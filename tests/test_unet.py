import torch

from satellite_diffusion_restoration.models import UNet


def test_unet_forward_shape_and_range():
    model = UNet(base_channels=4)
    x = torch.rand(2, 3, 64, 64)

    y = model(x)

    assert y.shape == x.shape
    y_detached = y.detach()
    assert torch.isfinite(y_detached).all()
    assert float(y_detached.min()) >= 0.0
    assert float(y_detached.max()) <= 1.0
