#!/usr/bin/env python
"""
Evaluation script for brain tumor segmentation.

Supports both vision-only and LLM-conditioned models.

Usage:
    # Vision-only
    python scripts/evaluate.py \
        --checkpoint outputs/baseline/best_model.pth \
        --config configs/baseline.yaml \
        --output_dir outputs/results

    # LLM-conditioned
    python scripts/evaluate.py \
        --checkpoint outputs/llm_conditioned/best_model.pth \
        --config configs/llm_conditioned.yaml \
        --output_dir outputs/results_llm
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from torch.cuda.amp import autocast
from tqdm import tqdm

from llmbrain.data import UCSFPDGMDataset, get_val_transforms
from llmbrain.data.dataset import prompt_collate_fn
from llmbrain.evaluation import compute_metrics, save_all_figures
from llmbrain.models import LLaMAEncoder, get_llm_conditioned_model, get_swin_unetr
from llmbrain.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--save_predictions", action="store_true", help="Save prediction NIfTI files")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    llm_config = config.get("llm", {})
    use_llm = llm_config.get("enabled", False)

    # Device
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    print("Loading evaluation data...")
    dataset = UCSFPDGMDataset(
        data_dir=data_config.get("data_dir", "./data/raw"),
        prompts_csv=data_config.get("prompts_csv"),
    )
    data_list = dataset.create_data_list()
    print(f"Found {len(data_list)} subjects")

    val_transforms = get_val_transforms()
    eval_dataset = Dataset(data=data_list, transform=val_transforms)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=prompt_collate_fn,
    )

    # ---- LLM Encoder ----
    llm_encoder = None
    text_dim = llm_config.get("text_dim", 3072)

    if use_llm:
        print("Loading LLaMA encoder...")
        llm_encoder = LLaMAEncoder(
            model_name=llm_config.get("model_name", "meta-llama/Llama-3.2-3B"),
            freeze=True,
            output_dim=llm_config.get("projection_dim"),
            pooling=llm_config.get("pooling", "last"),
            load_in_4bit=llm_config.get("load_in_4bit", True),
            load_in_8bit=llm_config.get("load_in_8bit", False),
        )
        text_dim = llm_encoder.output_dim

    # ---- Model ----
    print("Loading model...")
    if use_llm:
        model = get_llm_conditioned_model(
            img_size=tuple(model_config.get("img_size", [96, 96, 96])),
            in_channels=model_config.get("in_channels", 4),
            out_channels=model_config.get("out_channels", 4),
            feature_size=model_config.get("feature_size", 48),
            text_dim=text_dim,
            cross_attn_heads=model_config.get("cross_attn_heads", 8),
            cross_attn_dropout=model_config.get("cross_attn_dropout", 0.1),
            pretrained_vision=False,
            device=device,
        )
    else:
        model = get_swin_unetr(
            img_size=tuple(model_config.get("img_size", [96, 96, 96])),
            in_channels=model_config.get("in_channels", 4),
            out_channels=model_config.get("out_channels", 4),
            feature_size=model_config.get("feature_size", 48),
            pretrained=False,
            device=device,
        )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # ---- Evaluate ----
    print("Running evaluation...")
    model.eval()
    post_pred = AsDiscrete(argmax=True)

    predictions_dir = output_dir / "predictions"
    if args.save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = []
    sample_results = []

    with torch.no_grad():
        for batch_data in tqdm(eval_loader, desc="Evaluating"):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].numpy()
            subject_ids = batch_data.get("subject_id", ["unknown"])
            prompts = batch_data.get("prompt")

            # Encode text if LLM mode
            text_embeddings, text_mask = None, None
            if llm_encoder is not None and prompts is not None:
                text_embeddings, text_mask = llm_encoder.get_sequence_embeddings(prompts)

            # Build predictor
            if text_embeddings is not None:
                def _predict(x, te=text_embeddings, tm=text_mask):
                    return model(x, text_embeddings=te, text_mask=tm)
                predictor = _predict
            else:
                predictor = model

            with autocast():
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=(96, 96, 96),
                    sw_batch_size=4,
                    predictor=predictor,
                    overlap=0.5,
                )

            outputs_list = decollate_batch(outputs)
            predictions = [post_pred(o).cpu().numpy() for o in outputs_list]

            for i, (pred, label, subject_id) in enumerate(
                zip(predictions, labels, subject_ids)
            ):
                pred_squeezed = pred.squeeze()
                label_squeezed = label.squeeze()

                subject_metrics = compute_metrics(
                    pred_squeezed.astype(np.int32),
                    label_squeezed.astype(np.int32),
                    voxel_spacing=(1.0, 1.0, 1.0),
                )
                subject_metrics["subject_id"] = subject_id
                metrics_list.append(subject_metrics)

                if len(sample_results) < 10:
                    sample_results.append({
                        "subject_id": subject_id,
                        "image": inputs[i].cpu().numpy(),
                        "ground_truth": label_squeezed,
                        "prediction": pred_squeezed,
                    })

                if args.save_predictions:
                    pred_nii = nib.Nifti1Image(pred_squeezed.astype(np.int16), affine=np.eye(4))
                    nib.save(pred_nii, predictions_dir / f"{subject_id}_pred.nii.gz")

    # ---- Results ----
    metrics_df = pd.DataFrame(metrics_list)

    print("\n" + "=" * 60)
    mode_str = "LLM-CONDITIONED" if use_llm else "VISION-ONLY BASELINE"
    print(f"EVALUATION RESULTS ({mode_str})")
    print("=" * 60)

    metric_cols = ["dice_wt", "dice_tc", "dice_et", "hd95_wt", "hd95_tc", "hd95_et"]
    for col in metric_cols:
        if col in metrics_df.columns:
            values = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                print(f"{col.upper():12s}: {values.mean():.4f} +/- {values.std():.4f}")

    print("=" * 60)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nMetrics saved to {output_dir / 'metrics.csv'}")

    summary = {}
    for col in metric_cols:
        if col in metrics_df.columns:
            values = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                summary[col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                }

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(output_dir / "metrics_summary.csv")

    print("\nGenerating figures...")
    save_all_figures(output_dir, sample_results, metrics_df, num_samples=5)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
