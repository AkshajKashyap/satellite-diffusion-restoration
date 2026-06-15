"""Visualization helpers for restoration comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from satellite_diffusion_restoration.data.utils import to_numpy_image


def save_comparison_grid(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    restored: torch.Tensor | None = None,
    output_dir: str | Path = "outputs/samples",
    filename: str = "comparison.png",
) -> Path:
    """Save a clean/corrupted/restored comparison grid and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / filename

    images = [("Clean", clean), ("Corrupted", corrupted)]
    if restored is not None:
        images.append(("Restored", restored))

    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4), squeeze=False)
    for axis, (title, image) in zip(axes[0], images, strict=True):
        axis.imshow(to_numpy_image(image))
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(save_path, dpi=140)
    plt.close(figure)
    return save_path
