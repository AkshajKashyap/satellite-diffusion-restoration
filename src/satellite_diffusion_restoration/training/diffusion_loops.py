"""Training, sampling, and evaluation helpers for conditional DDPM restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from satellite_diffusion_restoration.evaluation.metrics import mse, psnr, ssim_batch
from satellite_diffusion_restoration.models import DDPMScheduler
from satellite_diffusion_restoration.training.loops import EvalMetrics


@dataclass
class DiffusionEvalResult:
    """Noise-prediction and sampled-restoration metrics."""

    noise_loss: float
    restoration: EvalMetrics


def train_diffusion_one_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> float:
    """Train a conditional DDPM noise predictor for one epoch."""
    criterion = loss_fn or nn.MSELoss()
    scheduler.to(device)
    model.train()

    total_loss = 0.0
    total_samples = 0
    for corrupted, clean in dataloader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)
        timesteps = torch.randint(
            0,
            scheduler.timesteps,
            (clean.size(0),),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean)
        noisy_clean = scheduler.q_sample(clean, timesteps, noise)

        optimizer.zero_grad(set_to_none=True)
        predicted_noise = model(noisy_clean, corrupted, timesteps)
        loss = criterion(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        batch_size = clean.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_noise_prediction(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> float:
    """Evaluate DDPM noise-prediction loss without running reverse sampling."""
    criterion = loss_fn or nn.MSELoss()
    scheduler.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    for corrupted, clean in dataloader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)
        timesteps = torch.randint(
            0,
            scheduler.timesteps,
            (clean.size(0),),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean)
        noisy_clean = scheduler.q_sample(clean, timesteps, noise)
        predicted_noise = model(noisy_clean, corrupted, timesteps)
        loss = criterion(predicted_noise, noise)

        batch_size = clean.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def sample_diffusion_restoration(
    model: nn.Module,
    scheduler: DDPMScheduler,
    corrupted: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample restored images from a conditional DDPM."""
    scheduler.to(device)
    corrupted = corrupted.to(device)
    return scheduler.sample(model, corrupted).cpu()


@torch.no_grad()
def evaluate_diffusion_restoration(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
    compute_ssim: bool = True,
) -> EvalMetrics:
    """Sample restorations and compare corrupted input vs DDPM output."""
    scheduler.to(device)
    model.eval()

    corrupted_inputs = []
    restored_images = []
    targets = []
    for corrupted, clean in dataloader:
        restored = sample_diffusion_restoration(model, scheduler, corrupted, device)
        corrupted_inputs.append(corrupted.cpu())
        restored_images.append(restored.cpu())
        targets.append(clean.cpu())

    corrupted_batch = torch.cat(corrupted_inputs, dim=0)
    restored_batch = torch.cat(restored_images, dim=0)
    target_batch = torch.cat(targets, dim=0)

    corrupted_ssim: float | None = None
    restored_ssim: float | None = None
    if compute_ssim:
        corrupted_ssim = ssim_batch(corrupted_batch, target_batch)
        restored_ssim = ssim_batch(restored_batch, target_batch)

    restored_mse = mse(restored_batch, target_batch)
    return EvalMetrics(
        loss=restored_mse,
        corrupted_mse=mse(corrupted_batch, target_batch),
        corrupted_psnr=psnr(corrupted_batch, target_batch),
        corrupted_ssim=corrupted_ssim,
        restored_mse=restored_mse,
        restored_psnr=psnr(restored_batch, target_batch),
        restored_ssim=restored_ssim,
    )


@torch.no_grad()
def evaluate_diffusion(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
    compute_ssim: bool = True,
) -> DiffusionEvalResult:
    """Evaluate both noise-prediction loss and sampled restoration metrics."""
    noise_loss = evaluate_noise_prediction(model, scheduler, dataloader, device)
    restoration = evaluate_diffusion_restoration(
        model,
        scheduler,
        dataloader,
        device,
        compute_ssim=compute_ssim,
    )
    return DiffusionEvalResult(noise_loss=noise_loss, restoration=restoration)
