#!/usr/bin/env python
"""
Training script for brain tumor segmentation.

Supports two modes:
  - Vision-only baseline:  python scripts/train.py --config configs/baseline.yaml
  - LLM-conditioned:       python scripts/train.py --config configs/llm_conditioned.yaml
"""

import argparse

import torch

from llmbrain.data import get_train_val_loaders
from llmbrain.models import LLaMAEncoder, get_llm_conditioned_model, get_swin_unetr
from llmbrain.training import Trainer, get_loss_function
from llmbrain.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. cuda:0, mps, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Config ----
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    llm_config = config.get("llm", {})
    training_config = config.get("training", {})
    output_dir = config.get("output_dir", "./outputs/baseline")

    use_llm = llm_config.get("enabled", False)

    # ---- Device ----
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---- Data loaders ----
    print("Creating data loaders...")
    train_loader, val_loader = get_train_val_loaders(
        data_dir=data_config.get("data_dir", "./data/raw"),
        prompts_csv=data_config.get("prompts_csv"),
        val_split=data_config.get("val_split", 0.2),
        batch_size=training_config.get("batch_size", 2),
        num_workers=training_config.get("num_workers", 4),
        img_size=tuple(model_config.get("img_size", [96, 96, 96])),
    )

    # ---- LLM Encoder (optional) ----
    llm_encoder = None
    text_dim = llm_config.get("text_dim", 3072)

    if use_llm:
        print("Loading LLaMA encoder...")
        llm_encoder = LLaMAEncoder(
            model_name=llm_config.get("model_name", "meta-llama/Llama-3.2-3B"),
            freeze=llm_config.get("freeze", True),
            output_dim=llm_config.get("projection_dim"),
            pooling=llm_config.get("pooling", "last"),
            load_in_4bit=llm_config.get("load_in_4bit", True),
            load_in_8bit=llm_config.get("load_in_8bit", False),
        )
        text_dim = llm_encoder.output_dim
        print(f"LLaMA encoder loaded (text_dim={text_dim}, frozen={llm_config.get('freeze', True)})")

    # ---- Model ----
    print("Creating segmentation model...")
    if use_llm:
        model = get_llm_conditioned_model(
            img_size=tuple(model_config.get("img_size", [96, 96, 96])),
            in_channels=model_config.get("in_channels", 4),
            out_channels=model_config.get("out_channels", 4),
            feature_size=model_config.get("feature_size", 48),
            text_dim=text_dim,
            cross_attn_heads=model_config.get("cross_attn_heads", 8),
            cross_attn_dropout=model_config.get("cross_attn_dropout", 0.1),
            pretrained_vision=model_config.get("pretrained", True),
            use_checkpoint=model_config.get("use_checkpoint", False),
            device=device,
        )
    else:
        model = get_swin_unetr(
            img_size=tuple(model_config.get("img_size", [96, 96, 96])),
            in_channels=model_config.get("in_channels", 4),
            out_channels=model_config.get("out_channels", 4),
            feature_size=model_config.get("feature_size", 48),
            pretrained=model_config.get("pretrained", True),
            use_checkpoint=model_config.get("use_checkpoint", False),
            device=device,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    # ---- Loss / Optimizer / Scheduler ----
    loss_fn = get_loss_function(
        loss_type="dice_ce",
        include_background=True,
        to_onehot_y=True,
        softmax=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 1e-5),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get("epochs", 300),
        eta_min=1e-6,
    )

    # ---- Resume ----
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        max_epochs=training_config.get("epochs", 300),
        val_interval=training_config.get("val_interval", 5),
        amp=training_config.get("amp", True),
        scheduler=scheduler,
        llm_encoder=llm_encoder,
    )

    trainer.train()


if __name__ == "__main__":
    main()
