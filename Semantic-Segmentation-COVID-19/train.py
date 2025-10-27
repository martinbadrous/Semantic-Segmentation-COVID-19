"""Training entry point for COVID-19 CT lesion segmentation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.unet import UNet
from utils.helpers import (
    create_dataloader,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_metrics,
    set_seed,
)
from utils.losses import get_loss
from utils.metrics import dice_coefficient, iou_score, pixel_accuracy


def _parse_features(raw: str) -> Sequence[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) < 2:
        raise argparse.ArgumentTypeError("Expected at least two integers for --features (e.g. '64,128,256,512')")
    return values


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
    train_loader = create_dataloader(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        image_mode=args.image_mode,
    )

    if args.val_images and args.val_masks:
        val_loader = create_dataloader(
            image_dir=args.val_images,
            mask_dir=args.val_masks,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            image_mode=args.image_mode,
        )
    else:
        val_loader = None
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    threshold: float,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_accuracy = 0.0
    num_samples = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            running_dice += dice_coefficient(probs, masks, threshold=threshold).item() * batch_size
            running_iou += iou_score(probs, masks, threshold=threshold).item() * batch_size
            running_accuracy += pixel_accuracy(probs, masks, threshold=threshold).item() * batch_size

        running_loss += loss.item() * batch_size
        num_samples += batch_size
        progress.set_postfix({"loss": loss.item()})

    return {
        "loss": running_loss / num_samples,
        "dice": running_dice / num_samples,
        "iou": running_iou / num_samples,
        "pixel_accuracy": running_accuracy / num_samples,
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_accuracy = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * batch_size
            running_dice += dice_coefficient(probs, masks, threshold=threshold).item() * batch_size
            running_iou += iou_score(probs, masks, threshold=threshold).item() * batch_size
            running_accuracy += pixel_accuracy(probs, masks, threshold=threshold).item() * batch_size
            num_samples += batch_size

    return {
        "loss": running_loss / num_samples,
        "dice": running_dice / num_samples,
        "iou": running_iou / num_samples,
        "pixel_accuracy": running_accuracy / num_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net for COVID-19 CT segmentation")
    parser.add_argument("--train-images", required=True, help="Directory with training CT slices")
    parser.add_argument("--train-masks", required=True, help="Directory with training masks")
    parser.add_argument("--val-images", help="Directory with validation CT slices")
    parser.add_argument("--val-masks", help="Directory with validation masks")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", type=str, default="bce_dice", help="Loss function to use")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE in BCE+Dice loss")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--features", type=_parse_features, default="64,128,256,512")
    parser.add_argument("--use-transposed-conv", action="store_true", help="Use transposed conv upsampling instead of bilinear")
    parser.add_argument("--image-mode", choices=["L", "RGB"], default="L", help="Color mode used when loading images")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume training from")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training when available")
    parser.add_argument("--val-threshold", type=float, default=0.5, help="Threshold for metric computation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--save-metrics", type=str, help="Optional path to save validation metrics as JSON")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 disables)")
    parser.add_argument("--patience-metric", choices=["dice", "iou", "pixel_accuracy"], default="dice")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(force_cpu=args.cpu)
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == "cuda" else None

    expected_channels = 3 if args.image_mode == "RGB" else 1
    if args.in_channels != expected_channels:
        print(
            f"[Info] Adjusting in_channels from {args.in_channels} to {expected_channels} to match image mode {args.image_mode}."
        )
        args.in_channels = expected_channels

    model_kwargs = {
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "features": args.features,
        "bilinear": not args.use_transposed_conv,
    }

    optimizer: torch.optim.Optimizer | None = None
    start_epoch = 1
    best_metric = float("-inf")
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        model_kwargs.update(checkpoint.model_args)
        best_metric = checkpoint.best_metric
        start_epoch = checkpoint.epoch + 1

    model = UNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if checkpoint is not None:
        model.load_state_dict(checkpoint.model_state)
        if checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)

    loss_kwargs: dict[str, float] = {}
    if args.loss.lower() in {"bce_dice", "bcedice", "bce+dice"}:
        loss_kwargs["bce_weight"] = args.bce_weight
    criterion = get_loss(args.loss, **loss_kwargs)

    train_loader, val_loader = build_dataloaders(args)
    patience_counter = 0

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        last_epoch = epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            threshold=args.val_threshold,
        )

        log_message = (
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_metrics['loss']:.4f}, "
            f"train_dice: {train_metrics['dice']:.4f}"
        )
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, threshold=args.val_threshold)
            log_message += f", val_dice: {val_metrics['dice']:.4f}, val_iou: {val_metrics['iou']:.4f}"

            metric_value = val_metrics[args.patience_metric]
            improved = metric_value > best_metric
            if improved:
                best_metric = metric_value
                patience_counter = 0
            else:
                patience_counter += 1

            if improved or val_loader is None:
                checkpoint_path = Path(args.checkpoint_dir) / "best.pt"
                save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_metric=best_metric,
                    model_args=model_kwargs,
                )

            if args.save_metrics:
                save_metrics(args.save_metrics, val_metrics)

            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        else:
            # If no validation set is provided, keep best metric as training dice
            metric_value = train_metrics["dice"]
            if metric_value > best_metric:
                best_metric = metric_value
                checkpoint_path = Path(args.checkpoint_dir) / "best.pt"
                save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_metric=best_metric,
                    model_args=model_kwargs,
                )

        print(log_message)

    final_path = Path(args.checkpoint_dir) / "last.pt"
    save_checkpoint(
        final_path,
        epoch=last_epoch,
        model=model,
        optimizer=optimizer,
        best_metric=best_metric,
        model_args=model_kwargs,
    )


if __name__ == "__main__":
    main()
