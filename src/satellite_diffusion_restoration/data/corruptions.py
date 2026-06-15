"""Seedable tensor corruptions for satellite image restoration smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


Range = tuple[float, float]
IntRange = tuple[int, int]


@dataclass(frozen=True)
class CorruptionConfig:
    """Configuration for the combined corruption transform."""

    noise_std: float = 0.05
    mask_probability: float = 0.7
    mask_count: IntRange = (1, 3)
    mask_size_fraction: Range = (0.12, 0.35)
    cloud_probability: float = 0.9
    cloud_count: IntRange = (2, 5)
    cloud_opacity: Range = (0.20, 0.65)
    cloud_radius_fraction: Range = (0.08, 0.28)
    blur_probability: float = 0.35
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0


def _validate_image(image: torch.Tensor) -> None:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise TypeError("Expected a floating point image tensor")


def _rand(generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    return torch.rand((), generator=generator, device=device)


def _randint(
    low: int,
    high: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    return int(torch.randint(low, high, (), generator=generator, device=device).item())


def _sample_int_range(
    value_range: IntRange,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    low, high = value_range
    if high < low:
        raise ValueError(f"Invalid integer range: {value_range}")
    return _randint(low, high + 1, generator, device)


def _sample_float_range(
    value_range: Range,
    generator: torch.Generator | None,
    device: torch.device,
) -> float:
    low, high = value_range
    if high < low:
        raise ValueError(f"Invalid float range: {value_range}")
    return float((low + (high - low) * _rand(generator, device)).item())


def _clamp_image(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0.0, 1.0)


def add_gaussian_noise(
    image: torch.Tensor,
    std: float = 0.05,
    mean: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise to a ``(C, H, W)`` image tensor in ``[0, 1]``."""
    _validate_image(image)
    if std <= 0:
        return _clamp_image(image)

    noise = torch.randn(
        image.shape,
        generator=generator,
        device=image.device,
        dtype=image.dtype,
    )
    return _clamp_image(image + noise * std + mean)


def add_random_rectangular_masks(
    image: torch.Tensor,
    num_masks: int | IntRange = (1, 3),
    size_fraction: Range = (0.12, 0.35),
    fill_value: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cover random rectangular regions with a constant value."""
    _validate_image(image)
    _, height, width = image.shape
    device = image.device

    if isinstance(num_masks, int):
        count = num_masks
    else:
        count = _sample_int_range(num_masks, generator, device)

    corrupted = image.clone()
    for _ in range(max(count, 0)):
        mask_h = max(1, int(height * _sample_float_range(size_fraction, generator, device)))
        mask_w = max(1, int(width * _sample_float_range(size_fraction, generator, device)))
        top = _randint(0, max(height - mask_h + 1, 1), generator, device)
        left = _randint(0, max(width - mask_w + 1, 1), generator, device)
        corrupted[:, top : top + mask_h, left : left + mask_w] = fill_value

    return _clamp_image(corrupted)


def add_soft_cloud_blobs(
    image: torch.Tensor,
    num_blobs: int | IntRange = (2, 5),
    opacity: Range = (0.20, 0.65),
    radius_fraction: Range = (0.08, 0.28),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Blend soft white cloud-like blobs over an image."""
    _validate_image(image)
    _, height, width = image.shape
    device = image.device
    dtype = image.dtype

    if isinstance(num_blobs, int):
        count = num_blobs
    else:
        count = _sample_int_range(num_blobs, generator, device)

    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, width)
    cloud_mask = torch.zeros((1, height, width), device=device, dtype=dtype)

    for _ in range(max(count, 0)):
        center_x = _sample_float_range((0.0, 1.0), generator, device)
        center_y = _sample_float_range((0.0, 1.0), generator, device)
        radius_x = _sample_float_range(radius_fraction, generator, device)
        radius_y = _sample_float_range(radius_fraction, generator, device)
        alpha = _sample_float_range(opacity, generator, device)

        blob = torch.exp(
            -(
                ((x - center_x) ** 2) / (2 * radius_x**2)
                + ((y - center_y) ** 2) / (2 * radius_y**2)
            )
        )
        cloud_mask = torch.maximum(cloud_mask, blob * alpha)

    return _clamp_image(image * (1.0 - cloud_mask) + cloud_mask)


def add_gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply a small channel-wise Gaussian blur."""
    _validate_image(image)
    if kernel_size <= 1 or sigma <= 0:
        return _clamp_image(image)
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    channels = image.shape[0]
    device = image.device
    dtype = image.dtype
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    blurred = F.conv2d(
        image.unsqueeze(0),
        kernel,
        padding=radius,
        groups=channels,
    ).squeeze(0)
    return _clamp_image(blurred)


class CombinedCorruption:
    """Apply noise, masks, soft clouds, and optional blur to a clean image."""

    def __init__(self, config: CorruptionConfig | None = None) -> None:
        self.config = config or CorruptionConfig()

    def __call__(
        self,
        image: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        _validate_image(image)
        config = self.config
        device = image.device

        corrupted = image.clone()

        if config.cloud_probability > 0 and _rand(generator, device) < config.cloud_probability:
            corrupted = add_soft_cloud_blobs(
                corrupted,
                num_blobs=config.cloud_count,
                opacity=config.cloud_opacity,
                radius_fraction=config.cloud_radius_fraction,
                generator=generator,
            )

        if config.mask_probability > 0 and _rand(generator, device) < config.mask_probability:
            corrupted = add_random_rectangular_masks(
                corrupted,
                num_masks=config.mask_count,
                size_fraction=config.mask_size_fraction,
                generator=generator,
            )

        corrupted = add_gaussian_noise(corrupted, std=config.noise_std, generator=generator)

        if config.blur_probability > 0 and _rand(generator, device) < config.blur_probability:
            corrupted = add_gaussian_blur(
                corrupted,
                kernel_size=config.blur_kernel_size,
                sigma=config.blur_sigma,
            )

        return _clamp_image(corrupted)
