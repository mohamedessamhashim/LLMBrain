"""Evaluation metrics for brain tumor segmentation."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """Compute Dice similarity coefficient.

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice score in range [0, 1].
    """
    intersection = np.sum(prediction * target)
    union = np.sum(prediction) + np.sum(target)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def hausdorff_distance_95(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute 95th percentile Hausdorff distance.

    Args:
        prediction: Binary prediction array.
        target: Binary ground truth array.
        voxel_spacing: Voxel spacing in mm.

    Returns:
        HD95 distance in mm. Returns inf if either mask is empty.
    """
    # Handle empty masks
    if np.sum(prediction) == 0 or np.sum(target) == 0:
        return float("inf")

    # Get surface points using distance transform
    pred_border = _get_surface_points(prediction)
    target_border = _get_surface_points(target)

    if len(pred_border) == 0 or len(target_border) == 0:
        return float("inf")

    # Scale by voxel spacing
    pred_border = pred_border * np.array(voxel_spacing)
    target_border = target_border * np.array(voxel_spacing)

    # Compute distances from prediction to target
    distances_pred_to_target = _compute_surface_distances(pred_border, target_border)

    # Compute distances from target to prediction
    distances_target_to_pred = _compute_surface_distances(target_border, pred_border)

    # Combine all distances
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])

    # Return 95th percentile
    return float(np.percentile(all_distances, 95))


def _get_surface_points(binary_mask: np.ndarray) -> np.ndarray:
    """Extract surface points from binary mask.

    Args:
        binary_mask: Binary mask array.

    Returns:
        Array of surface point coordinates.
    """
    # Erode the mask and XOR with original to get border
    from scipy.ndimage import binary_erosion

    eroded = binary_erosion(binary_mask)
    border = binary_mask ^ eroded

    # Get coordinates of border points
    points = np.array(np.where(border)).T

    return points


def _compute_surface_distances(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:
    """Compute minimum distances from points_a to points_b.

    Args:
        points_a: Source points (N, 3).
        points_b: Target points (M, 3).

    Returns:
        Array of minimum distances for each point in points_a.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points_b)
    distances, _ = tree.query(points_a)

    return distances


def compute_tumor_regions(segmentation: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute tumor region masks from segmentation.

    BraTS/UCSF-PDGM label convention:
        - 0: Background
        - 1: NCR (Necrotic tumor core)
        - 2: ED (Peritumoral edema)
        - 3 or 4: ET (GD-enhancing tumor)

    Tumor regions:
        - WT (Whole Tumor): NCR + ED + ET (labels 1, 2, 3/4)
        - TC (Tumor Core): NCR + ET (labels 1, 3/4)
        - ET (Enhancing Tumor): ET only (label 3/4)

    Args:
        segmentation: Integer segmentation array.

    Returns:
        Dictionary with binary masks for WT, TC, ET.
    """
    # Handle both BraTS (label 4 for ET) and UCSF-PDGM (label 3 for ET)
    ncr = (segmentation == 1)
    ed = (segmentation == 2)
    et = (segmentation == 3) | (segmentation == 4)

    regions = {
        "WT": (ncr | ed | et).astype(np.float32),  # Whole Tumor
        "TC": (ncr | et).astype(np.float32),       # Tumor Core
        "ET": et.astype(np.float32),               # Enhancing Tumor
    }

    return regions


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """Compute all evaluation metrics for brain tumor segmentation.

    Args:
        prediction: Predicted segmentation (integer labels).
        target: Ground truth segmentation (integer labels).
        voxel_spacing: Voxel spacing in mm for HD95.

    Returns:
        Dictionary containing Dice and HD95 for WT, TC, ET.
    """
    # Get tumor region masks
    pred_regions = compute_tumor_regions(prediction)
    target_regions = compute_tumor_regions(target)

    metrics = {}

    for region_name in ["WT", "TC", "ET"]:
        pred_mask = pred_regions[region_name]
        target_mask = target_regions[region_name]

        # Dice score
        dice = dice_score(pred_mask, target_mask)
        metrics[f"dice_{region_name.lower()}"] = dice

        # HD95
        hd95 = hausdorff_distance_95(pred_mask, target_mask, voxel_spacing)
        metrics[f"hd95_{region_name.lower()}"] = hd95

    return metrics


def aggregate_metrics(
    metrics_list: list,
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple subjects.

    Args:
        metrics_list: List of metrics dictionaries.

    Returns:
        Dictionary with mean and std for each metric.
    """
    if len(metrics_list) == 0:
        return {}

    # Get all metric names
    metric_names = list(metrics_list[0].keys())

    aggregated = {}
    for name in metric_names:
        values = [m[name] for m in metrics_list if not np.isinf(m[name])]
        if len(values) > 0:
            aggregated[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return aggregated
