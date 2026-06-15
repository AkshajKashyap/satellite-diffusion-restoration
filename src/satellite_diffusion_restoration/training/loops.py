"""Small supervised training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from satellite_diffusion_restoration.evaluation.metrics import mse, psnr, ssim_batch


@dataclass
class EvalMetrics:
    """Aggregate validation metrics."""

    loss: float
    mse: float
    psnr: float
    ssim: float | None


@dataclass
class History:
    """Minimal history tracker for scripts and checkpoints."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_psnr: list[float] = field(default_factory=list)
    val_ssim: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_psnr": self.val_psnr,
            "val_ssim": self.val_ssim,
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> float:
    """Train for one epoch and return mean loss."""
    criterion = loss_fn or nn.MSELoss()
    model.train()

    total_loss = 0.0
    total_samples = 0
    for corrupted, clean in dataloader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)

        optimizer.zero_grad(set_to_none=True)
        restored = model(corrupted)
        loss = criterion(restored, clean)
        loss.backward()
        optimizer.step()

        batch_size = corrupted.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    compute_ssim: bool = True,
) -> EvalMetrics:
    """Evaluate a model and return loss, MSE, PSNR, and optional SSIM."""
    criterion = loss_fn or nn.MSELoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    for corrupted, clean in dataloader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)
        restored = model(corrupted).clamp(0.0, 1.0)
        loss = criterion(restored, clean)

        batch_size = corrupted.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        predictions.append(restored.cpu())
        targets.append(clean.cpu())

    prediction_batch = torch.cat(predictions, dim=0)
    target_batch = torch.cat(targets, dim=0)

    ssim_score: float | None = None
    if compute_ssim:
        ssim_score = ssim_batch(prediction_batch, target_batch)

    return EvalMetrics(
        loss=total_loss / max(total_samples, 1),
        mse=mse(prediction_batch, target_batch),
        psnr=psnr(prediction_batch, target_batch),
        ssim=ssim_score,
    )
