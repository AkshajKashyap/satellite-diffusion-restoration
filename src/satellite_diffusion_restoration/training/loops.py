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
    corrupted_mse: float
    corrupted_psnr: float
    corrupted_ssim: float | None
    restored_mse: float
    restored_psnr: float
    restored_ssim: float | None

    @property
    def mse_delta(self) -> float:
        """Positive means the restored output reduced MSE vs the corrupted input."""
        return self.corrupted_mse - self.restored_mse

    @property
    def psnr_delta(self) -> float:
        """Positive means the restored output improved PSNR vs the corrupted input."""
        return self.restored_psnr - self.corrupted_psnr

    @property
    def mse(self) -> float:
        """Backward-compatible alias for restored MSE."""
        return self.restored_mse

    @property
    def psnr(self) -> float:
        """Backward-compatible alias for restored PSNR."""
        return self.restored_psnr

    @property
    def ssim(self) -> float | None:
        """Backward-compatible alias for restored SSIM."""
        return self.restored_ssim


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
    corrupted_inputs = []
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
        corrupted_inputs.append(corrupted.cpu())
        predictions.append(restored.cpu())
        targets.append(clean.cpu())

    corrupted_batch = torch.cat(corrupted_inputs, dim=0)
    prediction_batch = torch.cat(predictions, dim=0)
    target_batch = torch.cat(targets, dim=0)

    corrupted_ssim: float | None = None
    restored_ssim: float | None = None
    if compute_ssim:
        corrupted_ssim = ssim_batch(corrupted_batch, target_batch)
        restored_ssim = ssim_batch(prediction_batch, target_batch)

    return EvalMetrics(
        loss=total_loss / max(total_samples, 1),
        corrupted_mse=mse(corrupted_batch, target_batch),
        corrupted_psnr=psnr(corrupted_batch, target_batch),
        corrupted_ssim=corrupted_ssim,
        restored_mse=mse(prediction_batch, target_batch),
        restored_psnr=psnr(prediction_batch, target_batch),
        restored_ssim=restored_ssim,
    )
