"""A small, readable U-Net baseline for RGB image restoration."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two convolution, batch norm, ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downsample with max pooling, then apply a double conv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsample, concatenate a skip connection, then apply a double conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        height_delta = skip.size(2) - x.size(2)
        width_delta = skip.size(3) - x.size(3)
        if height_delta != 0 or width_delta != 0:
            x = F.pad(
                x,
                [
                    width_delta // 2,
                    width_delta - width_delta // 2,
                    height_delta // 2,
                    height_delta - height_delta // 2,
                ],
            )

        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Small supervised denoising U-Net for ``(N, 3, 64, 64)`` images."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 16,
        residual_mode: bool = True,
        residual_scale: float = 0.5,
        bound_residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual_mode = residual_mode
        self.residual_scale = residual_scale
        self.bound_residual = bound_residual

        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = Down(base_channels, base_channels * 2)
        self.encoder3 = Down(base_channels * 2, base_channels * 4)
        self.bottleneck = Down(base_channels * 4, base_channels * 8)

        self.decoder3 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = Up(base_channels * 2, base_channels, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        if self.residual_mode:
            self._initialize_identity_residual_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_image = x
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        x = self.bottleneck(skip3)

        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        output = self.output_conv(x)

        if not self.residual_mode:
            return torch.sigmoid(output)

        residual = output
        if self.bound_residual:
            residual = torch.tanh(residual) * self.residual_scale
        return (input_image + residual).clamp(0.0, 1.0)

    def predict_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned residual correction for debugging."""
        restored = self.forward(x)
        return restored - x

    def _initialize_identity_residual_head(self) -> None:
        """Start residual mode as an identity mapping: restored equals corrupted."""
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)
