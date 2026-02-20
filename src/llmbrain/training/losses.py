"""Loss functions for brain tumor segmentation."""

from typing import Optional

import torch
import torch.nn as nn
from monai.losses import DiceCELoss as MonaiDiceCELoss
from monai.losses import DiceLoss


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss.

    This loss combines soft Dice loss and cross-entropy loss,
    which is commonly used for medical image segmentation.

    Args:
        include_background: Whether to include background class in loss.
        to_onehot_y: Whether to convert targets to one-hot encoding.
        softmax: Whether to apply softmax to predictions.
        ce_weight: Optional class weights for cross-entropy.
        lambda_dice: Weight for Dice loss component.
        lambda_ce: Weight for cross-entropy component.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        softmax: bool = True,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 0.5,  # spec: L = L_Dice + 0.5*L_CE
    ):
        super().__init__()

        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

        self.loss_fn = MonaiDiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            weight=ce_weight,  # MONAI ≥1.4 renamed ce_weight → weight
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            predictions: Model predictions (B, C, H, W, D).
            targets: Ground truth labels (B, 1, H, W, D) or (B, C, H, W, D).

        Returns:
            Combined loss value.
        """
        return self.loss_fn(predictions, targets)


class DiceLossWrapper(nn.Module):
    """Wrapper for MONAI's Dice loss.

    Args:
        include_background: Whether to include background class.
        to_onehot_y: Whether to convert targets to one-hot.
        softmax: Whether to apply softmax to predictions.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        softmax: bool = True,
    ):
        super().__init__()

        self.loss_fn = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        return self.loss_fn(predictions, targets)


def get_loss_function(
    loss_type: str = "dice_ce",
    include_background: bool = True,
    to_onehot_y: bool = True,
    softmax: bool = True,
    ce_weight: Optional[torch.Tensor] = None,
    lambda_dice: float = 1.0,
    lambda_ce: float = 0.5,  # spec: L = L_Dice + 0.5*L_CE
) -> nn.Module:
    """Factory function to create loss function.

    Args:
        loss_type: Type of loss ("dice", "dice_ce").
        include_background: Whether to include background in loss.
        to_onehot_y: Whether to convert targets to one-hot.
        softmax: Whether to apply softmax to predictions.
        ce_weight: Optional class weights for CE.
        lambda_dice: Weight for Dice component.
        lambda_ce: Weight for CE component.

    Returns:
        Loss function module.
    """
    if loss_type == "dice":
        return DiceLossWrapper(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
        )
    elif loss_type == "dice_ce":
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            ce_weight=ce_weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
