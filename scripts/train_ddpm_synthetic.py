"""Train a small conditional DDPM restoration model on synthetic data."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/satellite_diffusion_restoration_matplotlib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested with --device cuda, but it is not available.")
    return torch.device(device_name)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from satellite_diffusion_restoration.data import CorruptionConfig
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import ConditionalUNet, DDPMScheduler
    from satellite_diffusion_restoration.training import (
        build_restoration_datasets,
        evaluate_diffusion,
        EMAModel,
        sample_diffusion_restoration,
        save_checkpoint,
        train_diffusion_one_epoch,
    )
    from satellite_diffusion_restoration.utils import seed_everything

    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_dataset, val_dataset = build_restoration_datasets(
        dataset_name="synthetic",
        image_size=64,
        seed=args.seed,
        max_train_samples=args.train_samples,
        max_val_samples=args.val_samples,
        corruption_config=CorruptionConfig(),
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConditionalUNet(base_channels=16, time_dim=64).to(device)
    scheduler = DDPMScheduler(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = None if args.disable_ema else EMAModel(model, decay=args.ema_decay)
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "ddpm_synthetic.pt"
    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    best_mse = float("inf")

    print("Synthetic conditional DDPM training")
    print(f"Device: {device}")
    print(
        "Config: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, lr={args.lr}, timesteps={args.timesteps}, "
        f"sampler={args.sampler}, sample_steps={args.sample_steps}, ema={ema is not None}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_diffusion_one_epoch(
            model,
            scheduler,
            train_loader,
            optimizer,
            device,
            ema=ema,
        )
        eval_model = _build_eval_model(model, ema, device)
        eval_result = evaluate_diffusion(
            eval_model,
            scheduler,
            val_loader,
            device,
            sampler=args.sampler,
            sample_steps=args.sample_steps,
        )
        metrics = eval_result.restoration

        if metrics.restored_mse < best_mse:
            best_mse = metrics.restored_mse
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "noise_loss": eval_result.noise_loss,
                    "corrupted_mse": metrics.corrupted_mse,
                    "corrupted_psnr": metrics.corrupted_psnr,
                    "restored_mse": metrics.restored_mse,
                    "restored_psnr": metrics.restored_psnr,
                    "mse_delta": metrics.mse_delta,
                    "psnr_delta": metrics.psnr_delta,
                },
                extra={
                    "dataset": "synthetic",
                    "model": "ConditionalUNet",
                    "base_channels": 16,
                    "time_dim": 64,
                    "scheduler": scheduler.state_dict(),
                    "sampler": args.sampler,
                    "sample_steps": args.sample_steps,
                    "ema_decay": args.ema_decay if ema is not None else None,
                    "seed": args.seed,
                },
                ema_state_dict=ema.state_dict() if ema is not None else None,
            )

        corrupted, clean = val_dataset[0]
        restored = sample_diffusion_restoration(
            eval_model,
            scheduler,
            corrupted.unsqueeze(0),
            device,
            sampler=args.sampler,
            sample_steps=args.sample_steps,
        ).squeeze(0)
        sample_path = save_comparison_grid(
            clean=clean,
            corrupted=corrupted,
            restored=restored,
            output_dir=sample_dir,
            filename=f"ddpm_synthetic_epoch_{epoch:02d}.png",
        )
        restored_ssim = "n/a" if metrics.restored_ssim is None else f"{metrics.restored_ssim:.4f}"
        corrupted_ssim = "n/a" if metrics.corrupted_ssim is None else f"{metrics.corrupted_ssim:.4f}"
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_noise_loss={train_loss:.6f} | val_noise_loss={eval_result.noise_loss:.6f} | "
            f"corrupted_mse={metrics.corrupted_mse:.6f} | ddpm_mse={metrics.restored_mse:.6f} | "
            f"mse_delta={metrics.mse_delta:+.6f} | "
            f"corrupted_psnr={metrics.corrupted_psnr:.2f} dB | "
            f"ddpm_psnr={metrics.restored_psnr:.2f} dB | psnr_delta={metrics.psnr_delta:+.2f} dB | "
            f"corrupted_ssim={corrupted_ssim} | ddpm_ssim={restored_ssim} | "
            f"sample={sample_path.relative_to(PROJECT_ROOT)}"
        )

    print(f"Best checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")


def _build_eval_model(model, ema, device):
    eval_model = copy.deepcopy(model).to(device)
    if ema is not None:
        ema.copy_to(eval_model)
    eval_model.eval()
    return eval_model


if __name__ == "__main__":
    main()
