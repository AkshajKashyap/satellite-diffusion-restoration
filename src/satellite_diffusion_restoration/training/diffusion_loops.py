"""Training, sampling, and evaluation helpers for conditional DDPM restoration."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict

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


class EMAModel:
    """Simple exponential moving average of model parameters for DDPM sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = decay
        self.shadow = OrderedDict(
            (name, parameter.detach().clone())
            for name, parameter in model.state_dict().items()
            if torch.is_floating_point(parameter)
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from a model."""
        model_state = model.state_dict()
        for name, shadow_value in self.shadow.items():
            shadow_value.mul_(self.decay).add_(model_state[name].detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights into a model in-place."""
        model_state = model.state_dict()
        for name, shadow_value in self.shadow.items():
            model_state[name].copy_(shadow_value)
        model.load_state_dict(model_state)

    def state_dict(self) -> dict[str, object]:
        """Return serializable EMA state."""
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, object], device: torch.device | str = "cpu") -> None:
        """Load EMA state."""
        self.decay = float(state_dict["decay"])
        shadow = state_dict["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("EMA shadow state must be a dictionary")
        self.shadow = OrderedDict((name, value.to(device)) for name, value in shadow.items())


def train_diffusion_one_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    ema: EMAModel | None = None,
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
        if ema is not None:
            ema.update(model)

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
    sampler: str = "ddpm",
    sample_steps: int | None = None,
) -> torch.Tensor:
    """Sample restored images from a conditional DDPM."""
    scheduler.to(device)
    corrupted = corrupted.to(device)
    return scheduler.sample(
        model,
        corrupted,
        sampler=sampler,
        sample_steps=sample_steps,
    ).cpu()


@torch.no_grad()
def evaluate_diffusion_restoration(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
    compute_ssim: bool = True,
    sampler: str = "ddpm",
    sample_steps: int | None = None,
) -> EvalMetrics:
    """Sample restorations and compare corrupted input vs DDPM output."""
    scheduler.to(device)
    model.eval()

    corrupted_inputs = []
    restored_images = []
    targets = []
    for corrupted, clean in dataloader:
        restored = sample_diffusion_restoration(
            model,
            scheduler,
            corrupted,
            device,
            sampler=sampler,
            sample_steps=sample_steps,
        )
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
def evaluate_one_step_x0_reconstruction(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
    timestep: int,
    compute_ssim: bool = True,
) -> EvalMetrics:
    """Evaluate one-step ``x0`` estimates from noised clean images at a fixed timestep."""
    scheduler.to(device)
    model.eval()
    clipped_timestep = min(max(timestep, 0), scheduler.timesteps - 1)

    corrupted_inputs = []
    estimated_images = []
    targets = []
    for corrupted, clean in dataloader:
        estimated_x0, _ = one_step_x0_diagnostic(
            model,
            scheduler,
            corrupted,
            clean,
            timestep=clipped_timestep,
            device=device,
        )
        corrupted_inputs.append(corrupted.cpu())
        estimated_images.append(estimated_x0.cpu())
        targets.append(clean.cpu())

    corrupted_batch = torch.cat(corrupted_inputs, dim=0)
    estimated_batch = torch.cat(estimated_images, dim=0)
    target_batch = torch.cat(targets, dim=0)

    corrupted_ssim: float | None = None
    estimated_ssim: float | None = None
    if compute_ssim:
        corrupted_ssim = ssim_batch(corrupted_batch, target_batch)
        estimated_ssim = ssim_batch(estimated_batch, target_batch)

    estimated_mse = mse(estimated_batch, target_batch)
    return EvalMetrics(
        loss=estimated_mse,
        corrupted_mse=mse(corrupted_batch, target_batch),
        corrupted_psnr=psnr(corrupted_batch, target_batch),
        corrupted_ssim=corrupted_ssim,
        restored_mse=estimated_mse,
        restored_psnr=psnr(estimated_batch, target_batch),
        restored_ssim=estimated_ssim,
    )


@torch.no_grad()
def evaluate_diffusion(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataloader: DataLoader,
    device: torch.device,
    compute_ssim: bool = True,
    sampler: str = "ddpm",
    sample_steps: int | None = None,
) -> DiffusionEvalResult:
    """Evaluate both noise-prediction loss and sampled restoration metrics."""
    noise_loss = evaluate_noise_prediction(model, scheduler, dataloader, device)
    restoration = evaluate_diffusion_restoration(
        model,
        scheduler,
        dataloader,
        device,
        compute_ssim=compute_ssim,
        sampler=sampler,
        sample_steps=sample_steps,
    )
    return DiffusionEvalResult(noise_loss=noise_loss, restoration=restoration)


@torch.no_grad()
def one_step_x0_diagnostic(
    model: nn.Module,
    scheduler: DDPMScheduler,
    corrupted: torch.Tensor,
    clean: torch.Tensor,
    timestep: int,
    device: torch.device,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-step ``x0`` estimate and the noised clean image used to make it."""
    scheduler.to(device)
    model.eval()
    corrupted = corrupted.to(device)
    clean = clean.to(device)
    timesteps = torch.full((clean.size(0),), timestep, device=device, dtype=torch.long)
    sampled_noise = torch.randn_like(clean) if noise is None else noise.to(device)
    noisy_clean = scheduler.q_sample(clean, timesteps, sampled_noise)
    predicted_noise = model(noisy_clean, corrupted, timesteps)
    estimated_x0 = scheduler.predict_x0_from_noise(noisy_clean, timesteps, predicted_noise)
    return estimated_x0.clamp(0.0, 1.0).cpu(), noisy_clean.cpu()


@torch.no_grad()
def compute_diffusion_diagnostics(
    model: nn.Module,
    scheduler: DDPMScheduler,
    corrupted: torch.Tensor,
    clean: torch.Tensor,
    device: torch.device,
    timesteps: tuple[int, ...] = (10, 50, 90),
    sampler: str = "ddim",
    sample_steps: int | None = 25,
    compute_ssim: bool = False,
) -> dict[str, object]:
    """Compute corrupted, one-step ``x0``, and full-sampling diagnostic metrics."""
    corrupted_cpu = corrupted.detach().cpu().clamp(0.0, 1.0)
    clean_cpu = clean.detach().cpu().clamp(0.0, 1.0)
    sampled = sample_diffusion_restoration(
        model,
        scheduler,
        corrupted,
        device,
        sampler=sampler,
        sample_steps=sample_steps,
    )

    one_step: dict[int, dict[str, float | None]] = {}
    for timestep in timesteps:
        clipped_timestep = min(max(timestep, 0), scheduler.timesteps - 1)
        estimated_x0, _ = one_step_x0_diagnostic(
            model,
            scheduler,
            corrupted,
            clean,
            timestep=clipped_timestep,
            device=device,
        )
        one_step[clipped_timestep] = _metric_dict(estimated_x0, clean_cpu, compute_ssim)

    return {
        "corrupted": _metric_dict(corrupted_cpu, clean_cpu, compute_ssim),
        "one_step": one_step,
        "sampled": _metric_dict(sampled, clean_cpu, compute_ssim),
    }


def _metric_dict(
    prediction: torch.Tensor,
    target: torch.Tensor,
    compute_ssim: bool,
) -> dict[str, float | None]:
    return {
        "mse": mse(prediction, target),
        "psnr": psnr(prediction, target),
        "ssim": ssim_batch(prediction, target) if compute_ssim else None,
    }
