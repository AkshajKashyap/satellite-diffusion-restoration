"""Train a tiny supervised U-Net denoising baseline on synthetic data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/satellite_diffusion_restoration_matplotlib")


def main() -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from satellite_diffusion_restoration.data import CorruptionConfig
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import UNet
    from satellite_diffusion_restoration.training import (
        History,
        build_synthetic_train_val_datasets,
        evaluate,
        get_device,
        save_checkpoint,
        train_one_epoch,
    )
    from satellite_diffusion_restoration.utils import seed_everything

    seed = 42
    seed_everything(seed)

    image_size = 64
    train_samples = 128
    val_samples = 32
    batch_size = 16
    epochs = 3
    learning_rate = 1e-3

    device = get_device(prefer_cuda=False)
    train_dataset, val_dataset = build_synthetic_train_val_datasets(
        train_samples=train_samples,
        val_samples=val_samples,
        image_size=image_size,
        seed=seed,
        corruption_config=CorruptionConfig(),
    )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(base_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history = History()
    best_val_loss = float("inf")
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "unet_baseline.pt"
    sample_dir = PROJECT_ROOT / "outputs" / "samples"

    print("U-Net baseline training")
    print(f"Device: {device}")
    print(
        "Config: "
        f"image_size={image_size}, train={train_samples}, val={val_samples}, "
        f"batch_size={batch_size}, epochs={epochs}, lr={learning_rate}"
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = evaluate(model, val_loader, device, loss_fn)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_metrics.loss)
        history.val_psnr.append(val_metrics.psnr)
        if val_metrics.ssim is not None:
            history.val_ssim.append(val_metrics.ssim)

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history.to_dict(),
                metrics={
                    "val_loss": val_metrics.loss,
                    "val_mse": val_metrics.mse,
                    "val_psnr": val_metrics.psnr,
                    "val_ssim": val_metrics.ssim if val_metrics.ssim is not None else float("nan"),
                },
                extra={
                    "model": "UNet",
                    "base_channels": 16,
                    "image_size": image_size,
                    "seed": seed,
                },
            )

        corrupted, clean = val_dataset[0]
        model.eval()
        with torch.no_grad():
            restored = model(corrupted.unsqueeze(0).to(device)).squeeze(0).cpu()
        sample_path = save_comparison_grid(
            clean=clean,
            corrupted=corrupted,
            restored=restored,
            output_dir=sample_dir,
            filename=f"unet_baseline_epoch_{epoch:02d}.png",
        )

        ssim_text = "n/a" if val_metrics.ssim is None else f"{val_metrics.ssim:.4f}"
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_metrics.loss:.6f} | "
            f"val_psnr={val_metrics.psnr:.2f} dB | val_ssim={ssim_text} | "
            f"sample={sample_path.relative_to(PROJECT_ROOT)}"
        )

    print(f"Best checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
