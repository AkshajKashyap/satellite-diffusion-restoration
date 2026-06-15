from pathlib import Path

import torch
from torch.utils.data import DataLoader

from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
from satellite_diffusion_restoration.models import UNet
from satellite_diffusion_restoration.training import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def test_tiny_training_step_runs():
    dataset = SyntheticSatelliteDataset(
        num_samples=2,
        image_size=32,
        seed=11,
        corruption_config=CorruptionConfig(cloud_probability=0.0, mask_probability=0.0),
    )
    dataloader = DataLoader(dataset, batch_size=2)
    model = UNet(base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, dataloader, optimizer, torch.device("cpu"))
    metrics = evaluate(model, dataloader, torch.device("cpu"), compute_ssim=False)

    assert loss > 0
    assert metrics.loss > 0
    assert metrics.ssim is None


def test_checkpoint_save_load_round_trip(tmp_path: Path):
    checkpoint_path = tmp_path / "model.pt"
    model = UNet(base_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        history={"train_loss": [0.2]},
        metrics={"val_psnr": 12.3},
    )

    loaded_model = UNet(base_channels=4)
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["history"]["train_loss"] == [0.2]
    for parameter, loaded_parameter in zip(model.parameters(), loaded_model.parameters(), strict=True):
        assert torch.equal(parameter, loaded_parameter)
