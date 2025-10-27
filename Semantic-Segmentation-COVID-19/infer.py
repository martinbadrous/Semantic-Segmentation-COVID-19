"""Inference script for generating lesion masks from CT slices."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from models.unet import UNet
from utils.helpers import create_dataloader, get_device, load_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a trained U-Net model")
    parser.add_argument("--images", required=True, help="Directory with images to segment")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--output", required=True, help="Directory where predicted masks will be stored")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference even when CUDA is available")
    parser.add_argument(
        "--save-probability-maps",
        action="store_true",
        help="Also persist the raw probability maps alongside binary masks",
    )
    return parser.parse_args()


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze().cpu().numpy()
    if array.ndim > 2:
        array = array[0]
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def run_inference(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model_args = {
        "in_channels": checkpoint.model_args.get("in_channels", 1),
        "out_channels": checkpoint.model_args.get("out_channels", 1),
        "features": checkpoint.model_args.get("features", (64, 128, 256, 512)),
        "bilinear": checkpoint.model_args.get("bilinear", True),
    }
    model = UNet(**model_args).to(device)
    model.load_state_dict(checkpoint.model_state)
    model.eval()

    image_mode = "RGB" if model_args["in_channels"] >= 3 else "L"
    dataloader = create_dataloader(
        image_dir=args.images,
        mask_dir=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        return_paths=True,
        image_mode=image_mode,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    if args.save_probability_maps:
        probability_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, paths in tqdm(dataloader, desc="Infer", leave=False):
            images = images.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > args.threshold).float()

            for prob, pred, path in zip(probabilities, predictions, paths):
                path = Path(path)
                mask_image = _tensor_to_image(pred)
                mask_path = output_dir / f"{path.stem}_mask.png"
                mask_image.save(mask_path)

                if args.save_probability_maps:
                    prob_image = _tensor_to_image(prob)
                    prob_path = probability_dir / f"{path.stem}_prob.png"
                    prob_image.save(prob_path)


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
