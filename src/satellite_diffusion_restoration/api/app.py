"""FastAPI app for residual U-Net satellite image restoration."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
import torch

from satellite_diffusion_restoration import __version__
from satellite_diffusion_restoration.inference import (
    DEFAULT_UNET_CHECKPOINT,
    MissingCheckpointError,
    pil_to_png_bytes,
    restore_image,
)


app = FastAPI(
    title="Satellite Diffusion Restoration",
    description="Residual U-Net restoration API for satellite image restoration under synthetic corruptions.",
    version=__version__,
)


def get_checkpoint_path() -> Path:
    """Resolve checkpoint path from environment or default."""
    env_path = os.environ.get("MODEL_CHECKPOINT") or os.environ.get(
        "SATELLITE_RESTORATION_CHECKPOINT"
    )
    return Path(env_path) if env_path else DEFAULT_UNET_CHECKPOINT


def get_device() -> torch.device:
    """Resolve model device from MODEL_DEVICE."""
    device_name = os.environ.get("MODEL_DEVICE", "auto")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


@app.get("/health")
def health() -> dict[str, object]:
    """Return service status and checkpoint availability."""
    checkpoint_path = get_checkpoint_path()
    return {
        "status": "ok",
        "model_type": "residual_unet",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "version": __version__,
    }


@app.post("/restore")
async def restore(file: UploadFile = File(...)) -> Response:
    """Restore an uploaded image and return PNG bytes."""
    checkpoint_path = get_checkpoint_path()
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc

    try:
        restored, _ = restore_image(image, checkpoint_path=checkpoint_path, device=get_device())
    except MissingCheckpointError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=pil_to_png_bytes(restored), media_type="image/png")
