"""Baseline image restoration metrics."""

from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


def mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean squared error between two tensors."""
    _validate_pair(prediction, target)
    return float(torch.mean((prediction.detach() - target.detach()) ** 2).item())


def psnr(prediction: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    """Return peak signal-to-noise ratio in decibels."""
    error = mse(prediction, target)
    if error == 0:
        return math.inf
    return 10.0 * math.log10((max_value**2) / error)


def ssim_batch(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average SSIM for ``(N, C, H, W)`` or ``(C, H, W)`` tensors."""
    _validate_pair(prediction, target)

    pred_batch = prediction.detach().cpu()
    target_batch = target.detach().cpu()
    if pred_batch.ndim == 3:
        pred_batch = pred_batch.unsqueeze(0)
        target_batch = target_batch.unsqueeze(0)
    if pred_batch.ndim != 4:
        raise ValueError("Expected tensors with shape (C, H, W) or (N, C, H, W)")

    scores = []
    for pred_image, target_image in zip(pred_batch, target_batch, strict=True):
        pred_np = _to_hwc(pred_image)
        target_np = _to_hwc(target_image)
        min_side = min(pred_np.shape[0], pred_np.shape[1])
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            raise ValueError("SSIM requires images at least 3x3 pixels")
        scores.append(
            structural_similarity(
                target_np,
                pred_np,
                channel_axis=-1,
                data_range=1.0,
                win_size=win_size,
            )
        )

    return float(np.mean(scores))


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Expected matching tensor shapes, got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Expected floating point tensors")


def _to_hwc(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")
    return image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
