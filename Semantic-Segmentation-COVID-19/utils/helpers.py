"""Utility helpers shared across training and inference scripts."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import CovidCTDataset


@dataclass(slots=True)
class Checkpoint:
    """A lightweight wrapper describing a saved model checkpoint."""

    epoch: int
    best_metric: float
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    model_args: dict[str, Any]


def set_seed(seed: int) -> None:
    """Sets seeds for Python, NumPy and PyTorch for reproducible experiments."""

    random.seed(seed)
    try:
        import numpy as np  # Imported lazily to keep dependency optional
    except ModuleNotFoundError:  # pragma: no cover - dependency optional
        np = None
    else:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    """Returns the preferred computation device."""

    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def create_dataloader(
    image_dir: str | Path,
    mask_dir: str | Path | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    transform: Any | None = None,
    return_paths: bool = False,
    image_mode: str = "L",
) -> DataLoader:
    """Utility that instantiates a :class:`~torch.utils.data.DataLoader`."""

    dataset = CovidCTDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        image_mode=image_mode,
        return_paths=return_paths,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    best_metric: float,
    model_args: dict[str, Any],
) -> None:
    """Persists a model checkpoint to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_args": model_args,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> Checkpoint:
    """Loads a checkpoint from disk and returns a :class:`Checkpoint`."""

    payload = torch.load(path, map_location=map_location)
    return Checkpoint(
        epoch=int(payload.get("epoch", 0)),
        best_metric=float(payload.get("best_metric", 0.0)),
        model_state=dict(payload["model_state"]),
        optimizer_state=payload.get("optimizer_state"),
        model_args=dict(payload.get("model_args", {})),
    )


def save_metrics(path: str | Path, metrics: dict[str, float]) -> None:
    """Writes metrics to a JSON file to simplify experiment tracking."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


__all__ = [
    "Checkpoint",
    "set_seed",
    "get_device",
    "create_dataloader",
    "save_checkpoint",
    "load_checkpoint",
    "save_metrics",
]
