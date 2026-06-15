"""Small image tensor utilities shared by datasets and visualizations."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy_image(image: torch.Tensor) -> np.ndarray:
    """Convert a ``(C, H, W)`` tensor in ``[0, 1]`` to a matplotlib RGB array."""
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")
    if image.shape[0] not in {1, 3}:
        raise ValueError("Expected 1 or 3 image channels")

    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array
