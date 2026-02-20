"""Visualization utilities for brain tumor segmentation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Color map for tumor regions (matching BraTS convention)
TUMOR_COLORS = {
    0: [0, 0, 0],         # Background (black)
    1: [255, 0, 0],       # NCR - Necrotic core (red)
    2: [0, 255, 0],       # ED - Edema (green)
    3: [255, 255, 0],     # ET - Enhancing tumor (yellow)
    4: [255, 255, 0],     # ET - Alternative label (yellow)
}


def segmentation_to_rgb(segmentation: np.ndarray) -> np.ndarray:
    """Convert segmentation labels to RGB image.

    Args:
        segmentation: Integer segmentation array (H, W) or (H, W, D).

    Returns:
        RGB array with shape (..., 3).
    """
    rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)

    for label, color in TUMOR_COLORS.items():
        mask = segmentation == label
        rgb[mask] = color

    return rgb


def plot_triplanar_view(
    image: np.ndarray,
    segmentation: Optional[np.ndarray] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot triplanar view (axial, coronal, sagittal) of 3D volume.

    Args:
        image: 3D image array (H, W, D) or (C, H, W, D).
        segmentation: Optional segmentation overlay.
        slice_indices: Tuple of (axial, coronal, sagittal) slice indices.
        title: Figure title.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Handle multi-channel images (use first channel or FLAIR)
    if image.ndim == 4:
        image = image[0]  # Use first channel

    # Default slice indices at center
    if slice_indices is None:
        slice_indices = (
            image.shape[0] // 2,
            image.shape[1] // 2,
            image.shape[2] // 2,
        )

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Normalize image for display
    vmin, vmax = np.percentile(image[image > 0], [1, 99]) if np.any(image > 0) else (0, 1)

    # Axial slice
    ax_slice = image[slice_indices[0], :, :]
    axes[0].imshow(ax_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title(f"Axial (z={slice_indices[0]})")

    # Coronal slice
    cor_slice = image[:, slice_indices[1], :]
    axes[1].imshow(cor_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Coronal (y={slice_indices[1]})")

    # Sagittal slice
    sag_slice = image[:, :, slice_indices[2]]
    axes[2].imshow(sag_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title(f"Sagittal (x={slice_indices[2]})")

    # Overlay segmentation if provided
    if segmentation is not None:
        alpha = 0.5

        # Axial
        seg_ax = segmentation_to_rgb(segmentation[slice_indices[0], :, :])
        mask_ax = segmentation[slice_indices[0], :, :] > 0
        seg_overlay = np.zeros((*ax_slice.shape, 4))
        seg_overlay[..., :3] = seg_ax / 255.0
        seg_overlay[..., 3] = mask_ax * alpha
        axes[0].imshow(seg_overlay.transpose(1, 0, 2), origin="lower")

        # Coronal
        seg_cor = segmentation_to_rgb(segmentation[:, slice_indices[1], :])
        mask_cor = segmentation[:, slice_indices[1], :] > 0
        seg_overlay = np.zeros((*cor_slice.shape, 4))
        seg_overlay[..., :3] = seg_cor / 255.0
        seg_overlay[..., 3] = mask_cor * alpha
        axes[1].imshow(seg_overlay.transpose(1, 0, 2), origin="lower")

        # Sagittal
        seg_sag = segmentation_to_rgb(segmentation[:, :, slice_indices[2]])
        mask_sag = segmentation[:, :, slice_indices[2]] > 0
        seg_overlay = np.zeros((*sag_slice.shape, 4))
        seg_overlay[..., :3] = seg_sag / 255.0
        seg_overlay[..., 3] = mask_sag * alpha
        axes[2].imshow(seg_overlay.transpose(1, 0, 2), origin="lower")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_comparison_grid(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    title: str = "",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison grid: MRI | Ground Truth | Prediction.

    Args:
        image: 3D image array.
        ground_truth: Ground truth segmentation.
        prediction: Predicted segmentation.
        slice_idx: Slice index to display.
        axis: Axis for slicing (0=axial, 1=coronal, 2=sagittal).
        title: Figure title.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Handle multi-channel images
    if image.ndim == 4:
        image = image[0]

    # Default to center slice
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2

    # Get slices
    slices = [slice(None)] * 3
    slices[axis] = slice_idx

    img_slice = image[tuple(slices)]
    gt_slice = ground_truth[tuple(slices)]
    pred_slice = prediction[tuple(slices)]

    # Normalize image
    vmin, vmax = np.percentile(img_slice[img_slice > 0], [1, 99]) if np.any(img_slice > 0) else (0, 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # MRI
    axes[0].imshow(img_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("MRI")

    # Ground Truth
    axes[1].imshow(img_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    gt_rgb = segmentation_to_rgb(gt_slice)
    mask_gt = gt_slice > 0
    overlay = np.zeros((*img_slice.shape, 4))
    overlay[..., :3] = gt_rgb / 255.0
    overlay[..., 3] = mask_gt * 0.6
    axes[1].imshow(overlay.transpose(1, 0, 2), origin="lower")
    axes[1].set_title("Ground Truth")

    # Prediction
    axes[2].imshow(img_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    pred_rgb = segmentation_to_rgb(pred_slice)
    mask_pred = pred_slice > 0
    overlay = np.zeros((*img_slice.shape, 4))
    overlay[..., :3] = pred_rgb / 255.0
    overlay[..., 3] = mask_pred * 0.6
    axes[2].imshow(overlay.transpose(1, 0, 2), origin="lower")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_dice_boxplot(
    metrics_df,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot boxplot of Dice scores for each tumor region.

    Args:
        metrics_df: DataFrame with columns dice_wt, dice_tc, dice_et.
        save_path: Optional path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for boxplot
    dice_cols = ["dice_wt", "dice_tc", "dice_et"]
    labels = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]

    data = [metrics_df[col].dropna() for col in dice_cols if col in metrics_df.columns]
    present_labels = [labels[i] for i, col in enumerate(dice_cols) if col in metrics_df.columns]

    # Create boxplot
    bp = ax.boxplot(data, labels=present_labels, patch_artist=True)

    # Colors
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors[: len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Dice Scores by Tumor Region", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add mean markers
    means = [d.mean() for d in data]
    for i, mean in enumerate(means):
        ax.scatter(i + 1, mean, marker="D", color="white", s=50, zorder=3, edgecolors="black")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def save_all_figures(
    output_dir: str,
    sample_results: List[Dict],
    metrics_df,
    num_samples: int = 5,
) -> None:
    """Save all evaluation figures.

    Args:
        output_dir: Output directory for figures.
        sample_results: List of dicts with image, ground_truth, prediction, subject_id.
        metrics_df: DataFrame with per-subject metrics.
        num_samples: Number of sample comparison figures to save.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save Dice boxplot
    plot_dice_boxplot(metrics_df, save_path=figures_dir / "dice_boxplot.png")
    plt.close()

    # Save sample comparisons
    for i, result in enumerate(sample_results[:num_samples]):
        subject_id = result.get("subject_id", f"sample_{i}")

        # Triplanar view
        plot_triplanar_view(
            result["image"],
            result["prediction"],
            title=f"{subject_id} - Prediction",
            save_path=figures_dir / f"{subject_id}_triplanar.png",
        )
        plt.close()

        # Comparison grid
        plot_comparison_grid(
            result["image"],
            result["ground_truth"],
            result["prediction"],
            title=subject_id,
            save_path=figures_dir / f"{subject_id}_comparison.png",
        )
        plt.close()

    print(f"Figures saved to {figures_dir}")
