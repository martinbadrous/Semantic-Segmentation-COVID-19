"""Dataset utilities for loading paired CT images and masks."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["CovidCTDataset"]


class CovidCTDataset(Dataset):
    """Dataset that loads image/mask pairs for segmentation.

    The dataset expects two directories containing images and masks with identical
    filenames. Masks can be omitted (``mask_dir=None``) when running inference.
    """

    SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        transform: callable | None = None,
        image_mode: str = "L",
        return_paths: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform
        self.return_paths = return_paths
        self.image_mode = image_mode

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory '{self.image_dir}' does not exist")

        self.image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images with extensions {self.SUPPORTED_EXTENSIONS} found in {self.image_dir}")

        if self.mask_dir is not None:
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"Mask directory '{self.mask_dir}' does not exist")
            mask_paths = {
                p.stem: p
                for p in self.mask_dir.iterdir()
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            }
            if not mask_paths:
                raise FileNotFoundError(
                    f"No masks with extensions {self.SUPPORTED_EXTENSIONS} found in {self.mask_dir}"
                )
            self.matched_masks = []
            for image_path in self.image_paths:
                mask_path = mask_paths.get(image_path.stem)
                if mask_path is None:
                    raise FileNotFoundError(
                        f"Could not find a mask named '{image_path.stem}' in '{self.mask_dir}'"
                    )
                self.matched_masks.append(mask_path)
        else:
            self.matched_masks = None

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _load_image(path: Path, mode: str = "RGB") -> Image.Image:
        with Image.open(path) as img:
            return img.convert(mode)

    @staticmethod
    def _pil_to_tensor(image: Image.Image, is_mask: bool) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., None]
        if not is_mask:
            array = array / 255.0
        else:
            array = (array > 0).astype(np.float32)
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        if is_mask:
            tensor = tensor.clamp_(0, 1)
        return tensor

    def __getitem__(self, index: int):  # type: ignore[override]
        image_path = self.image_paths[index]
        image = self._load_image(image_path, mode=self.image_mode)

        mask_tensor = None
        if self.matched_masks is not None:
            mask_path = self.matched_masks[index]
            mask = self._load_image(mask_path, mode="L")
        else:
            mask = None

        if self.transform is not None:
            if mask is not None:
                try:
                    transformed = self.transform(image=image, mask=mask)
                except TypeError:
                    transformed = self.transform(image, mask)
                if isinstance(transformed, dict):
                    image = transformed.get("image", image)
                    mask = transformed.get("mask", mask)
                elif isinstance(transformed, (tuple, list)):
                    image, mask = transformed
                else:
                    image = transformed
            else:
                try:
                    transformed = self.transform(image=image)
                except TypeError:
                    transformed = self.transform(image)
                image = transformed.get("image", transformed) if isinstance(transformed, dict) else transformed

        if isinstance(image, Image.Image):
            image_tensor = self._pil_to_tensor(image, is_mask=False)
        elif isinstance(image, torch.Tensor):
            image_tensor = image.float()
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.ndim == 3 and image_tensor.max() > 1:
                image_tensor = image_tensor / 255.0
        else:
            array = np.asarray(image, dtype=np.float32)
            if array.ndim == 2:
                array = array[..., None]
            array = array / 255.0
            image_tensor = torch.from_numpy(array.transpose(2, 0, 1))

        if mask is not None:
            if isinstance(mask, Image.Image):
                mask_tensor = self._pil_to_tensor(mask, is_mask=True)
            elif isinstance(mask, torch.Tensor):
                mask_tensor = mask.float()
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                if mask_tensor.max() > 1:
                    mask_tensor = mask_tensor / 255.0
                mask_tensor = (mask_tensor > 0.5).float()
            else:
                array = np.asarray(mask, dtype=np.float32)
                if array.ndim == 2:
                    array = array[..., None]
                mask_tensor = torch.from_numpy((array > 0).astype(np.float32).transpose(2, 0, 1))

        if self.return_paths:
            if mask_tensor is None:
                return image_tensor, str(image_path)
            return image_tensor, mask_tensor, str(image_path)

        if mask_tensor is None:
            return image_tensor
        return image_tensor, mask_tensor
