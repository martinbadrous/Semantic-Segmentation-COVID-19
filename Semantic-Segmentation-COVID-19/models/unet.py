"""Implementation of a lightweight U-Net architecture used for segmentation."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 block used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial wrapper
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net implementation with configurable encoder width."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] | Iterable[int] = (64, 128, 256, 512),
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.features = tuple(int(f) for f in features)
        if len(self.features) < 2:
            raise ValueError("`features` must contain at least two stages")

        self.bilinear = bilinear
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in self.features:
            self.downs.append(DoubleConv(current_channels, feature))
            current_channels = feature

        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * factor)

        reversed_features = list(reversed(self.features))
        prev_channels = self.features[-1] * factor
        for feature in reversed_features:
            if bilinear:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        nn.Conv2d(prev_channels, feature, kernel_size=1),
                    )
                )
            else:
                self.ups.append(nn.ConvTranspose2d(prev_channels, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
            prev_channels = feature

        self.out_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            upsample = self.ups[idx]
            conv = self.ups[idx + 1]
            x = upsample(x)
            skip = skip_connections[idx // 2]

            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.size(2) - x.size(2)
                diff_x = skip.size(3) - x.size(3)
                x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.out_conv(x)


__all__ = ["UNet"]
