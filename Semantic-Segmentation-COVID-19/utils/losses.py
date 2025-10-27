"""Loss functions used for training the segmentation models."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _flatten_binary_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattens predictions and targets for binary segmentation losses."""

    if logits.shape != targets.shape:
        raise ValueError("Logits and targets must have the same shape for Dice loss computation")

    return logits.contiguous().view(logits.size(0), -1), targets.contiguous().view(targets.size(0), -1)


class DiceLoss(nn.Module):
    """Soft Dice loss operating on logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat, targets_flat = _flatten_binary_logits(probs, targets)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        denominator = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice_score.mean()


class BCEDiceLoss(nn.Module):
    """Combination of binary cross entropy and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if not 0 <= bce_weight <= 1:
            raise ValueError("`bce_weight` must be between 0 and 1")
        self.bce_weight = bce_weight
        self.dice_weight = 1 - bce_weight
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_loss(name: str, **kwargs: object) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Factory returning a segmentation loss by name."""

    normalized = name.lower()
    if normalized in {"dice", "diceloss"}:
        return DiceLoss(**kwargs)
    if normalized in {"bce_dice", "bcedice", "bce+dice"}:
        return BCEDiceLoss(**kwargs)
    if normalized in {"bce", "binary_cross_entropy"}:
        return nn.BCEWithLogitsLoss(**kwargs)
    raise ValueError(f"Unknown loss '{name}'. Available options: 'dice', 'bce', 'bce_dice'.")


__all__ = ["DiceLoss", "BCEDiceLoss", "get_loss"]
