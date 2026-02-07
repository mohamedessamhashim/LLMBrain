"""Training loop for LLM-conditioned brain tumor segmentation."""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.amp = amp
        self.scheduler = scheduler
        self.llm_encoder = llm_encoder

        # Setup output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Mixed precision
        self.scaler = GradScaler() if amp else None

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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        step = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}")
        for batch_data in pbar:
            step += 1
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            prompts = batch_data.get("prompt")

            # Encode clinical text
            text_embeddings, text_mask = self._encode_prompts(prompts)

            self.optimizer.zero_grad()

            if self.amp:
                with autocast():
                    if text_embeddings is not None:
                        outputs = self.model(
                            inputs,
                            text_embeddings=text_embeddings,
                            text_mask=text_mask,
                        )
                    else:
                        outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if text_embeddings is not None:
                    outputs = self.model(
                        inputs,
                        text_embeddings=text_embeddings,
                        text_mask=text_mask,
                    )
                else:
                    outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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

                # Build predictor for sliding window inference
                if text_embeddings is not None:
                    # Closure captures the text embeddings for this sample
                    def _predict(x, te=text_embeddings, tm=text_mask):
                        return self.model(x, text_embeddings=te, text_mask=tm)
                    predictor = _predict
                else:
                    predictor = self.model

                if self.amp:
                    with autocast():
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
