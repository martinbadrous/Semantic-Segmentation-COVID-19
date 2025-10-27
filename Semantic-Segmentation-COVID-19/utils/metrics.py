"""Segmentation metrics used for model evaluation."""
from __future__ import annotations

import torch

_EPS = 1e-7


def _prepare_predictions(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure predictions and targets are binarized and aligned."""

    if preds.shape != targets.shape:
        raise ValueError("Predictions and targets must share the same shape for metric computation")

    if preds.dtype.is_floating_point:
        preds = preds > threshold
    else:
        preds = preds.bool()

    if targets.dtype.is_floating_point:
        targets = targets > 0.5
    else:
        targets = targets.bool()
    return preds, targets


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Computes the Sørensen–Dice coefficient for binary segmentation."""

    preds, targets = _prepare_predictions(preds, targets, threshold=threshold)
    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    denom = preds.sum(dim=(1, 2, 3)).float() + targets.sum(dim=(1, 2, 3)).float()
    dice = (2 * intersection + _EPS) / (denom + _EPS)
    return dice.mean()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Computes the Intersection over Union (Jaccard) score."""

    preds, targets = _prepare_predictions(preds, targets, threshold=threshold)
    intersection = (preds & targets).sum(dim=(1, 2, 3)).float()
    union = (preds | targets).sum(dim=(1, 2, 3)).float()
    return ((intersection + _EPS) / (union + _EPS)).mean()


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Computes the proportion of correctly classified pixels."""

    preds, targets = _prepare_predictions(preds, targets, threshold=threshold)
    correct = (preds == targets).float().sum(dim=(1, 2, 3))
    total = torch.tensor(preds[0].numel(), device=preds.device, dtype=torch.float32)
    return (correct / total).mean()


def evaluate_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Returns a dictionary with commonly used segmentation metrics."""

    with torch.no_grad():
        dice = dice_coefficient(preds, targets, threshold=threshold)
        iou = iou_score(preds, targets, threshold=threshold)
        acc = pixel_accuracy(preds, targets, threshold=threshold)
    return {"dice": float(dice.item()), "iou": float(iou.item()), "pixel_accuracy": float(acc.item())}


__all__ = ["dice_coefficient", "iou_score", "pixel_accuracy", "evaluate_metrics"]
