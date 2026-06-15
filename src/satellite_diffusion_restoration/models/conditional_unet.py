"""Conditional U-Net noise predictor for DDPM restoration."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings with shape ``(N, embedding_dim)``."""
    half_dim = embedding_dim // 2
    device = timesteps.device
    exponent = -math.log(10_000.0) * torch.arange(half_dim, device=device) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class TimeDoubleConv(nn.Module):
    """Two conv blocks with an additive timestep projection."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = x + self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x)


class ConditionalDown(nn.Module):
    """Downsample, then apply a time-aware double conv block."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = TimeDoubleConv(in_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x), time_embedding)


class ConditionalUp(nn.Module):
    """Upsample, concatenate skip features, then apply a time-aware double conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = TimeDoubleConv(out_channels + skip_channels, out_channels, time_dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
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
        return self.conv(torch.cat([skip, x], dim=1), time_embedding)


class ConditionalUNet(nn.Module):
    """Predict DDPM noise from ``x_t``, corrupted conditioning image, and timestep."""

    def __init__(
        self,
        image_channels: int = 3,
        condition_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 16,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        input_channels = image_channels + condition_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.encoder1 = TimeDoubleConv(input_channels, base_channels, time_dim)
        self.encoder2 = ConditionalDown(base_channels, base_channels * 2, time_dim)
        self.encoder3 = ConditionalDown(base_channels * 2, base_channels * 4, time_dim)
        self.bottleneck = ConditionalDown(base_channels * 4, base_channels * 8, time_dim)
        self.decoder3 = ConditionalUp(base_channels * 8, base_channels * 4, base_channels * 4, time_dim)
        self.decoder2 = ConditionalUp(base_channels * 4, base_channels * 2, base_channels * 2, time_dim)
        self.decoder1 = ConditionalUp(base_channels * 2, base_channels, base_channels, time_dim)
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        corrupted: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.shape != corrupted.shape:
            raise ValueError(
                f"Expected matching x_t and corrupted shapes, got {x_t.shape} and {corrupted.shape}"
            )

        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_dim)
        time_embedding = self.time_mlp(time_embedding)
        x = torch.cat([x_t, corrupted], dim=1)

        skip1 = self.encoder1(x, time_embedding)
        skip2 = self.encoder2(skip1, time_embedding)
        skip3 = self.encoder3(skip2, time_embedding)
        x = self.bottleneck(skip3, time_embedding)
        x = self.decoder3(x, skip3, time_embedding)
        x = self.decoder2(x, skip2, time_embedding)
        x = self.decoder1(x, skip1, time_embedding)
        return self.output_conv(x)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
