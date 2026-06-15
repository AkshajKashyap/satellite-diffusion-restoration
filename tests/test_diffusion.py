import torch
from torch.utils.data import DataLoader

from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
from satellite_diffusion_restoration.models import (
    ConditionalUNet,
    DDPMScheduler,
    sinusoidal_timestep_embedding,
)
from satellite_diffusion_restoration.training import sample_diffusion_restoration, train_diffusion_one_epoch
from satellite_diffusion_restoration.training import compute_diffusion_diagnostics, EMAModel


def test_scheduler_q_sample_shape_and_finite_values():
    scheduler = DDPMScheduler(timesteps=8)
    clean = torch.rand(2, 3, 16, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 7], dtype=torch.long)

    noisy = scheduler.q_sample(clean, timesteps, noise)

    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()
    assert float(noisy.abs().max()) < 5.0


def test_predict_x0_from_noise_recovers_clean_image():
    scheduler = DDPMScheduler(timesteps=8)
    clean = torch.rand(2, 3, 16, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([3, 7], dtype=torch.long)
    noisy = scheduler.q_sample(clean, timesteps, noise)

    estimated = scheduler.predict_x0_from_noise(noisy, timesteps, noise)

    assert estimated.shape == clean.shape
    assert torch.isfinite(estimated).all()
    assert torch.allclose(estimated, clean, atol=1e-5)


def test_timestep_embedding_shape_and_finite_values():
    timesteps = torch.tensor([0, 1, 5], dtype=torch.long)

    embedding = sinusoidal_timestep_embedding(timesteps, embedding_dim=15)

    assert embedding.shape == (3, 15)
    assert torch.isfinite(embedding).all()


def test_conditional_unet_forward_shape():
    model = ConditionalUNet(base_channels=4, time_dim=16)
    noisy = torch.rand(2, 3, 16, 16)
    corrupted = torch.rand(2, 3, 16, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.long)

    predicted_noise = model(noisy, corrupted, timesteps)

    assert predicted_noise.shape == noisy.shape
    assert torch.isfinite(predicted_noise).all()


def test_tiny_diffusion_train_step_runs():
    dataset = SyntheticSatelliteDataset(
        num_samples=2,
        image_size=16,
        seed=21,
        corruption_config=CorruptionConfig(cloud_probability=0.0, mask_probability=0.0),
    )
    dataloader = DataLoader(dataset, batch_size=2)
    model = ConditionalUNet(base_channels=4, time_dim=16)
    scheduler = DDPMScheduler(timesteps=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_diffusion_one_epoch(
        model,
        scheduler,
        dataloader,
        optimizer,
        torch.device("cpu"),
    )

    assert loss > 0


def test_ema_update_does_not_crash():
    model = ConditionalUNet(base_channels=4, time_dim=16)
    ema = EMAModel(model, decay=0.9)

    for parameter in model.parameters():
        parameter.data.add_(0.1)
    ema.update(model)
    ema.copy_to(model)

    assert ema.state_dict()["decay"] == 0.9


def test_diffusion_sampling_returns_image_range():
    model = ConditionalUNet(base_channels=4, time_dim=16)
    scheduler = DDPMScheduler(timesteps=4)
    corrupted = torch.rand(2, 3, 16, 16)

    restored = sample_diffusion_restoration(
        model,
        scheduler,
        corrupted,
        torch.device("cpu"),
    )

    assert restored.shape == corrupted.shape
    assert torch.isfinite(restored).all()
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_ddim_sampling_returns_image_range():
    model = ConditionalUNet(base_channels=4, time_dim=16)
    scheduler = DDPMScheduler(timesteps=8)
    corrupted = torch.rand(2, 3, 16, 16)

    restored = sample_diffusion_restoration(
        model,
        scheduler,
        corrupted,
        torch.device("cpu"),
        sampler="ddim",
        sample_steps=4,
    )

    assert restored.shape == corrupted.shape
    assert torch.isfinite(restored).all()
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_diagnostic_metric_structure_contains_expected_keys():
    model = ConditionalUNet(base_channels=4, time_dim=16)
    scheduler = DDPMScheduler(timesteps=8)
    corrupted = torch.rand(1, 3, 16, 16)
    clean = torch.rand(1, 3, 16, 16)

    diagnostics = compute_diffusion_diagnostics(
        model,
        scheduler,
        corrupted,
        clean,
        torch.device("cpu"),
        timesteps=(1, 4),
        sampler="ddim",
        sample_steps=4,
    )

    assert set(diagnostics) == {"corrupted", "one_step", "sampled"}
    assert set(diagnostics["one_step"]) == {1, 4}
    assert {"mse", "psnr", "ssim"} <= set(diagnostics["corrupted"])
    assert {"mse", "psnr", "ssim"} <= set(diagnostics["sampled"])
