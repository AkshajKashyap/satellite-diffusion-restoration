from io import BytesIO
import asyncio

from fastapi import HTTPException
from PIL import Image
import torch

from satellite_diffusion_restoration.api.app import app, health, restore
from satellite_diffusion_restoration.inference import (
    MissingCheckpointError,
    load_unet_checkpoint,
    preprocess_image,
    tensor_to_pil,
)


class FakeUpload:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def read(self) -> bytes:
        return self.payload


def test_inference_preprocessing_shape_and_range():
    image = Image.new("RGB", (20, 24), color=(120, 80, 40))

    tensor = preprocess_image(image, image_size=64)

    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_inference_postprocessing_returns_pil_image():
    tensor = torch.rand(3, 16, 16)

    image = tensor_to_pil(tensor)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_missing_checkpoint_error_is_clear(tmp_path):
    missing_checkpoint = tmp_path / "missing.pt"

    try:
        load_unet_checkpoint(missing_checkpoint)
    except MissingCheckpointError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected MissingCheckpointError")

    assert "Missing U-Net checkpoint" in message
    assert "python scripts/train_unet_eurosat.py --download" in message


def test_fastapi_health_endpoint(monkeypatch, tmp_path):
    missing_checkpoint = tmp_path / "missing.pt"
    monkeypatch.setenv("SATELLITE_RESTORATION_CHECKPOINT", str(missing_checkpoint))

    payload = health()

    assert payload["status"] == "ok"
    assert payload["model_type"] == "residual_unet"
    assert payload["checkpoint_exists"] is False


def test_fastapi_routes_are_registered():
    paths = {route.path for route in app.routes}

    assert "/health" in paths
    assert "/restore" in paths


def test_fastapi_restore_missing_checkpoint(monkeypatch, tmp_path):
    missing_checkpoint = tmp_path / "missing.pt"
    monkeypatch.setenv("SATELLITE_RESTORATION_CHECKPOINT", str(missing_checkpoint))
    image = Image.new("RGB", (16, 16), color=(50, 90, 120))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    upload = FakeUpload(buffer.getvalue())

    try:
        asyncio.run(restore(upload))
    except HTTPException as exc:
        error = exc
    else:
        raise AssertionError("Expected HTTPException")

    assert error.status_code == 503
    assert "Missing U-Net checkpoint" in error.detail
