"""MONAI transforms for brain tumor segmentation."""

from typing import List, Tuple

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Spacingd,
)

# UCSF-PDGM follows BraTS 2021 convention: ET is label 4 (not 3).
# The model uses 4 output classes {0,1,2,3}. We must remap label 4 → 3
# so CrossEntropyLoss does not crash with "Target 4 is out of bounds".
_LABEL_REMAP = MapLabelValued(
    keys=["label"],
    orig_labels=[0, 1, 2, 4],
    target_labels=[0, 1, 2, 3],
)


def get_train_transforms(
    img_size: Tuple[int, int, int] = (96, 96, 96),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Get training transforms with data augmentation.

    Args:
        img_size: Target patch size for random cropping.
        pixdim: Target voxel spacing (already 1mm isotropic after preprocessing).

    Returns:
        MONAI Compose transform for training.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Remap BraTS ET label 4 → model class 3 (4-class output).
            # UCSF-PDGM uses BraTS 2021 labels {0,1,2,4}; model expects {0,1,2,3}.
            _LABEL_REMAP,
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=img_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Get validation/inference transforms without augmentation.

    Args:
        pixdim: Target voxel spacing.

    Returns:
        MONAI Compose transform for validation/inference.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Remap BraTS ET label 4 → model class 3 (4-class output).
            _LABEL_REMAP,
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_inference_transforms(
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Get transforms for inference (no labels).

    Args:
        pixdim: Target voxel spacing.

    Returns:
        MONAI Compose transform for inference.
    """
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"]),
        ]
    )
