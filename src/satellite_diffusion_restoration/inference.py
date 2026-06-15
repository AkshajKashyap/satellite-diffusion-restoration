"""Inference helpers for the deployable residual U-Net restoration baseline."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/satellite_diffusion_restoration_matplotlib")

import numpy as np
from PIL import Image
import torch

from satellite_diffusion_restoration.data.eurosat import image_to_rgb_tensor
from satellite_diffusion_restoration.evaluation.metrics import mse, psnr, ssim_batch
from satellite_diffusion_restoration.models import UNet
from satellite_diffusion_restoration.training import load_checkpoint


DEFAULT_UNET_CHECKPOINT = Path("outputs/checkpoints/unet_eurosat.pt")
TRAIN_UNET_COMMAND = "python scripts/train_unet_eurosat.py --download"


class MissingCheckpointError(FileNotFoundError):
    """Raised when the deployable U-Net checkpoint is missing."""


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    """Return the checkpoint path, honoring an optional explicit override."""
    if checkpoint_path is not None:
        return Path(checkpoint_path)

    env_path = os.environ.get("MODEL_CHECKPOINT")
    return Path(env_path) if env_path else DEFAULT_UNET_CHECKPOINT


def load_unet_checkpoint(
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> UNet:
    """Load the residual U-Net restoration checkpoint."""
    path = resolve_checkpoint_path(checkpoint_path)
    if not path.exists():
        raise MissingCheckpointError(
            f"Missing U-Net checkpoint: {path}. Train it with `{TRAIN_UNET_COMMAND}`."
        )

    checkpoint_preview = torch.load(path, map_location=device)
    extra: dict[str, Any] = checkpoint_preview.get("extra", {})
    model = UNet(
        base_channels=extra.get("base_channels", 16),
        residual_mode=extra.get("residual_mode", True),
        residual_scale=extra.get("residual_scale", 0.5),
    ).to(device)
    load_checkpoint(path, model=model, device=device)
    model.eval()
    return model


def preprocess_image(image: Image.Image | torch.Tensor | np.ndarray, image_size: int = 64) -> torch.Tensor:
    """Convert an image-like object to a ``(3, H, W)`` tensor in ``[0, 1]``."""
    return image_to_rgb_tensor(image, image_size=image_size)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a ``(C, H, W)`` image tensor in ``[0, 1]`` to a PIL RGB image."""
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(image.shape)}")
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGB")


def pil_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@torch.no_grad()
def restore_tensor(
    image: torch.Tensor,
    model: UNet | None = None,
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Restore a single image tensor with the residual U-Net."""
    model = model or load_unet_checkpoint(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    input_tensor = image.unsqueeze(0).to(device)
    restored = model(input_tensor).squeeze(0).cpu().clamp(0.0, 1.0)
    return restored


def restore_image(
    image: Image.Image | torch.Tensor | np.ndarray,
    model: UNet | None = None,
    checkpoint_path: str | Path | None = None,
    device: torch.device | str = "cpu",
    image_size: int = 64,
    clean_target: Image.Image | torch.Tensor | np.ndarray | None = None,
) -> tuple[Image.Image, dict[str, float] | None]:
    """Restore an image-like object and optionally compute metrics against a clean target."""
    input_tensor = preprocess_image(image, image_size=image_size)
    restored_tensor = restore_tensor(
        input_tensor,
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    metrics = None
    if clean_target is not None:
        clean_tensor = preprocess_image(clean_target, image_size=image_size)
        metrics = compute_restoration_metrics(restored_tensor, clean_tensor)
    return tensor_to_pil(restored_tensor), metrics


def compute_restoration_metrics(restored: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    """Compute MSE, PSNR, and SSIM for restored and clean image tensors."""
    return {
        "mse": mse(restored, clean),
        "psnr": psnr(restored, clean),
        "ssim": ssim_batch(restored, clean),
    }
