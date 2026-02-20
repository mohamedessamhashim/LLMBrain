"""Training loop for LLM-conditioned brain tumor segmentation."""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_GENERIC_PROMPT = "brain MRI"


class Trainer:
    """Training loop supporting both vision-only and LLM-conditioned models.

    When an ``llm_encoder`` is provided the trainer feeds clinical prompts
    through the LLM to obtain sequence embeddings and passes them to the
    segmentation model via cross-attention.  When ``llm_encoder`` is *None*
    the trainer falls back to standard vision-only training.

    Args:
        model: Segmentation network (SwinUNETRBaseline or LLMConditionedSwinUNETR).
        loss_fn: Loss function.
        optimizer: Optimizer (should already exclude frozen LLM params).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device for training.
        output_dir: Checkpoint / log directory.
        max_epochs: Training epochs.
        val_interval: Validate every N epochs.
        amp: Automatic mixed precision.
        scheduler: Optional LR scheduler.
        llm_encoder: Optional LLaMA encoder for text conditioning.
        accumulation_steps: Gradient accumulation steps (effective batch = batch_size × steps).
        grad_clip: Max gradient norm for clipping (None = disabled).
        phase2_epoch: Epoch at which to unfreeze Swin encoder (two-phase schedule).
                      Set to None to skip two-phase training (e.g. for baseline).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        output_dir: str = "./outputs",
        max_epochs: int = 300,
        val_interval: int = 5,
        amp: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        llm_encoder: Optional[nn.Module] = None,
        prompt_dropout: float = 0.15,
        prompt_permutation: bool = True,
        accumulation_steps: int = 1,
        grad_clip: Optional[float] = 1.0,
        phase2_epoch: Optional[int] = 50,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        # device_type strips index suffix (e.g. "cuda:0" → "cuda") for autocast/GradScaler
        self.device_type = device.split(":")[0]
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.amp = amp
        self.scheduler = scheduler
        self.llm_encoder = llm_encoder
        self.prompt_dropout = prompt_dropout
        self.prompt_permutation = prompt_permutation
        self.accumulation_steps = max(1, accumulation_steps)
        self.grad_clip = grad_clip
        self.phase2_epoch = phase2_epoch

        # Setup output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Mixed precision (GradScaler and autocast both require device type, not full name)
        self.scaler = GradScaler(self.device_type) if amp else None

        # Metrics
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.post_pred = AsDiscrete(argmax=True, to_onehot=4)
        self.post_label = AsDiscrete(to_onehot=4)

        # Best metric tracking
        self.best_metric = -1
        self.best_epoch = -1

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _encode_prompts(self, prompts: List[str]):
        """Encode clinical prompts via the LLM.

        Returns (text_embeddings, text_mask) or (None, None) when no LLM is set.
        """
        if self.llm_encoder is None or prompts is None:
            return None, None

        text_embeddings, text_mask = self.llm_encoder.get_sequence_embeddings(prompts)
        return text_embeddings, text_mask

    def _augment_prompts(self, prompts: List[str]) -> List[str]:
        """Apply prompt augmentation during training.

        - Prompt dropout: replace with generic fallback with probability
          ``self.prompt_dropout`` to prevent over-reliance on text.
        - Field permutation: randomly shuffle the comma-separated clinical
          fields to prevent position-dependent attention patterns.
        """
        if prompts is None:
            return prompts

        augmented = []
        for prompt in prompts:
            # Prompt dropout
            if random.random() < self.prompt_dropout:
                augmented.append(_GENERIC_PROMPT)
                continue

            # Field permutation
            if self.prompt_permutation:
                fields = [f.strip() for f in prompt.split(",")]
                random.shuffle(fields)
                prompt = ", ".join(fields)

            augmented.append(prompt)
        return augmented

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Two-phase training schedule
    # ------------------------------------------------------------------

    def _set_phase(self, epoch: int) -> None:
        """Manage two-phase training schedule.

        Phase 1 (epoch < phase2_epoch): Swin encoder frozen, only adapters trained.
        Phase 2 (epoch >= phase2_epoch): All parameters unfrozen.

        This protects pretrained Swin weights during early adapter warm-up and
        then allows full fine-tuning once the cross-attention layers are stable.
        """
        if self.phase2_epoch is None:
            return  # No two-phase schedule

        in_phase2 = epoch >= self.phase2_epoch

        for name, param in self.model.named_parameters():
            if "swin_unetr.swinViT" in name or (
                "swin_unetr.encoder" in name and "cross_attn" not in name
            ):
                param.requires_grad = in_phase2

        if epoch == 0 and not in_phase2:
            print(f"[Two-phase] Phase 1: Swin encoder frozen until epoch {self.phase2_epoch}")
        elif epoch == self.phase2_epoch:
            print("[Two-phase] Phase 2: Swin encoder unfrozen — fine-tuning all parameters")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and clipping."""
        self.model.train()
        epoch_loss = 0.0
        step = 0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            step += 1
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            prompts = batch_data.get("prompt")

            # Augment and encode clinical text
            prompts = self._augment_prompts(prompts)
            text_embeddings, text_mask = self._encode_prompts(prompts)

            # Scale loss by accumulation steps so gradients average correctly
            if self.amp:
                with autocast(self.device_type):
                    if text_embeddings is not None:
                        outputs = self.model(
                            inputs,
                            text_embeddings=text_embeddings,
                            text_mask=text_mask,
                        )
                    else:
                        outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels) / self.accumulation_steps

                self.scaler.scale(loss).backward()
            else:
                if text_embeddings is not None:
                    outputs = self.model(
                        inputs,
                        text_embeddings=text_embeddings,
                        text_mask=text_mask,
                    )
                else:
                    outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels) / self.accumulation_steps
                loss.backward()

            # Optimizer step after accumulation_steps batches (or at end of epoch)
            is_last = batch_idx == num_batches - 1
            if (step % self.accumulation_steps == 0) or is_last:
                if self.amp:
                    # Unscale before clipping so clip threshold is in original scale
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=self.grad_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            max_norm=self.grad_clip,
                        )
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Track un-scaled loss for logging
            epoch_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})

        epoch_loss /= step
        return epoch_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                prompts = batch_data.get("prompt")

                text_embeddings, text_mask = self._encode_prompts(prompts)

                # Build predictor for sliding window inference.
                # Expand text embeddings to match the sw_batch_size that
                # sliding_window_inference may use internally.
                if text_embeddings is not None:
                    def _predict(x, te=text_embeddings, tm=text_mask):
                        te_exp = te.expand(x.shape[0], -1, -1)
                        tm_exp = tm.expand(x.shape[0], -1) if tm is not None else None
                        return self.model(x, text_embeddings=te_exp, text_mask=tm_exp)
                    predictor = _predict
                else:
                    predictor = self.model

                if self.amp:
                    with autocast(self.device_type):
                        outputs = sliding_window_inference(
                            inputs,
                            roi_size=(96, 96, 96),
                            sw_batch_size=4,
                            predictor=predictor,
                            overlap=0.5,
                        )
                else:
                    outputs = sliding_window_inference(
                        inputs,
                        roi_size=(96, 96, 96),
                        sw_batch_size=4,
                        predictor=predictor,
                        overlap=0.5,
                    )

                # Post-process predictions and labels
                outputs_list = decollate_batch(outputs)
                labels_list = decollate_batch(labels)

                outputs_post = [self.post_pred(o) for o in outputs_list]
                labels_post = [self.post_label(l) for l in labels_list]

                self.dice_metric(y_pred=outputs_post, y=labels_post)

        # Compute metrics
        dice_scores = self.dice_metric.aggregate()

        metrics = {
            "dice_bg": dice_scores[0].item(),
            "dice_ncr": dice_scores[1].item(),
            "dice_ed": dice_scores[2].item(),
            "dice_et": dice_scores[3].item(),
        }
        metrics["dice_mean"] = dice_scores[1:].mean().item()

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.output_dir / "latest_model.pth")

        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pth")
            print(f"New best model saved with mean Dice: {metrics['dice_mean']:.4f}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        mode = "LLM-conditioned" if self.llm_encoder is not None else "vision-only"
        print(f"Starting {mode} training for {self.max_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.max_epochs):
            # Apply two-phase schedule: freeze/unfreeze Swin encoder as needed
            self._set_phase(epoch)

            train_loss = self.train_epoch(epoch)

            self.writer.add_scalar("train/loss", train_loss, epoch)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Train Loss: {train_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
                self.writer.add_scalar(
                    "train/lr", self.scheduler.get_last_lr()[0], epoch
                )

            if (epoch + 1) % self.val_interval == 0:
                metrics = self.validate(epoch)

                for name, value in metrics.items():
                    self.writer.add_scalar(f"val/{name}", value, epoch)

                print(
                    f"Validation - "
                    f"Dice Mean: {metrics['dice_mean']:.4f}, "
                    f"NCR: {metrics['dice_ncr']:.4f}, "
                    f"ED: {metrics['dice_ed']:.4f}, "
                    f"ET: {metrics['dice_et']:.4f}"
                )

                is_best = metrics["dice_mean"] > self.best_metric
                if is_best:
                    self.best_metric = metrics["dice_mean"]
                    self.best_epoch = epoch + 1

                self.save_checkpoint(epoch, metrics, is_best)

        print(f"\nTraining complete!")
        print(f"Best validation Dice: {self.best_metric:.4f} at epoch {self.best_epoch}")

        self.writer.close()
