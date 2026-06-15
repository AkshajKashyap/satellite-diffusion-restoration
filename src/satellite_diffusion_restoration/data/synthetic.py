"""Synthetic pseudo-satellite dataset for tests and smoke runs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch.utils.data import Dataset

from satellite_diffusion_restoration.data.corruptions import CombinedCorruption, CorruptionConfig


class SyntheticSatelliteDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return deterministic ``(corrupted, clean)`` image pairs."""

    def __init__(
        self,
        num_samples: int = 8,
        image_size: int | tuple[int, int] = 64,
        seed: int = 0,
        corruption_config: CorruptionConfig | dict[str, Any] | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.height, self.width = _normalize_image_size(image_size)
        self.seed = seed

        if isinstance(corruption_config, dict):
            config = CorruptionConfig(**corruption_config)
        else:
            config = corruption_config or CorruptionConfig()

        self.corruption_config = config
        self.corrupt = CombinedCorruption(config)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)

        clean_generator = torch.Generator().manual_seed(self.seed + index)
        corrupt_generator = torch.Generator().manual_seed(self.seed + 10_000 + index)

        clean = _make_clean_image(self.height, self.width, clean_generator)
        corrupted = self.corrupt(clean, generator=corrupt_generator)
        return corrupted, clean

    def config_dict(self) -> dict[str, Any]:
        """Return a serializable description of the dataset setup."""
        return {
            "num_samples": self.num_samples,
            "image_size": (self.height, self.width),
            "seed": self.seed,
            "corruption_config": asdict(self.corruption_config),
        }


def _normalize_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size

    if height <= 0 or width <= 0:
        raise ValueError("image_size must be positive")
    return height, width


def _rand(generator: torch.Generator) -> torch.Tensor:
    return torch.rand((), generator=generator)


def _randint(low: int, high: int, generator: torch.Generator) -> int:
    return int(torch.randint(low, high, (), generator=generator).item())


def _make_clean_image(height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, 1, width)

    base = torch.cat(
        [
            0.16 + 0.10 * torch.sin(2.5 * torch.pi * x) + 0.04 * torch.cos(3.0 * torch.pi * y),
            0.33 + 0.10 * torch.cos(2.0 * torch.pi * y) + 0.05 * torch.sin(4.0 * torch.pi * x),
            0.18 + 0.08 * torch.sin(2.0 * torch.pi * (x + y)),
        ],
        dim=0,
    ).expand(3, height, width)

    image = base.clone()
    _paint_rectangles(image, generator)
    _paint_circles(image, generator)

    texture = torch.randn((3, height, width), generator=generator) * 0.025
    return (image + texture).clamp(0.0, 1.0)


def _paint_rectangles(image: torch.Tensor, generator: torch.Generator) -> None:
    _, height, width = image.shape
    palette = torch.tensor(
        [
            [0.17, 0.42, 0.16],
            [0.45, 0.37, 0.18],
            [0.28, 0.50, 0.24],
            [0.34, 0.31, 0.28],
        ],
        dtype=image.dtype,
    )

    for _ in range(_randint(5, 10, generator)):
        rect_h = _randint(max(3, height // 10), max(4, height // 3), generator)
        rect_w = _randint(max(3, width // 10), max(4, width // 3), generator)
        top = _randint(0, max(1, height - rect_h + 1), generator)
        left = _randint(0, max(1, width - rect_w + 1), generator)
        color = palette[_randint(0, len(palette), generator)].view(3, 1, 1)
        strength = 0.45 + float(_rand(generator).item()) * 0.35
        image[:, top : top + rect_h, left : left + rect_w] = (
            image[:, top : top + rect_h, left : left + rect_w] * (1.0 - strength)
            + color * strength
        )


def _paint_circles(image: torch.Tensor, generator: torch.Generator) -> None:
    _, height, width = image.shape
    y = torch.linspace(0.0, 1.0, height).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, 1, width)
    colors = torch.tensor(
        [
            [0.07, 0.18, 0.30],
            [0.12, 0.34, 0.12],
            [0.55, 0.55, 0.50],
        ],
        dtype=image.dtype,
    )

    for _ in range(_randint(3, 7, generator)):
        center_x = float(_rand(generator).item())
        center_y = float(_rand(generator).item())
        radius = 0.04 + float(_rand(generator).item()) * 0.16
        mask = (((x - center_x) ** 2 + (y - center_y) ** 2) < radius**2).to(image.dtype)
        color = colors[_randint(0, len(colors), generator)].view(3, 1, 1)
        image[:] = image * (1.0 - mask * 0.45) + color * mask * 0.45
