"""Train a conditional DDPM restoration model on EuroSAT RGB images."""

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
    parser.add_argument("--download", action="store_true", help="Download EuroSAT if missing.")
    parser.add_argument("--root", default="data/raw", help="EuroSAT root directory.")
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-val-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=32)
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
    from torch.utils.data import Subset
    from torch.utils.data import DataLoader

    from satellite_diffusion_restoration.data import CorruptionConfig
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import ConditionalUNet, DDPMScheduler
    from satellite_diffusion_restoration.training import (
        build_restoration_datasets,
        EMAModel,
        evaluate_diffusion,
        sample_diffusion_restoration,
        save_checkpoint,
        train_diffusion_one_epoch,
    )
    from satellite_diffusion_restoration.utils import seed_everything

    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    try:
        train_dataset, val_dataset = build_restoration_datasets(
            dataset_name="eurosat",
            root=args.root,
            download=args.download,
            image_size=64,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            corruption_config=CorruptionConfig(),
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nRun with download enabled:\n"
            "  python scripts/train_ddpm_eurosat.py --download"
        ) from exc

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    sampled_eval_count = min(args.eval_samples, len(val_dataset))
    sampled_eval_dataset = Subset(val_dataset, range(sampled_eval_count))
    sampled_eval_loader = DataLoader(
        sampled_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ConditionalUNet(base_channels=16, time_dim=64).to(device)
    scheduler = DDPMScheduler(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = None if args.disable_ema else EMAModel(model, decay=args.ema_decay)
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "ddpm_eurosat.pt"
    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    best_mse = float("inf")

    print("EuroSAT conditional DDPM training", flush=True)
    print(f"Device: {device}", flush=True)
    print(
        "Config: "
        f"root={args.root}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, "
        f"timesteps={args.timesteps}, sampled_eval={sampled_eval_count}, "
        f"sampler={args.sampler}, sample_steps={args.sample_steps}, ema={ema is not None}",
        flush=True,
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
            sampled_eval_loader,
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
                    "corrupted_ssim": (
                        metrics.corrupted_ssim if metrics.corrupted_ssim is not None else float("nan")
                    ),
                    "restored_mse": metrics.restored_mse,
                    "restored_psnr": metrics.restored_psnr,
                    "restored_ssim": (
                        metrics.restored_ssim if metrics.restored_ssim is not None else float("nan")
                    ),
                    "mse_delta": metrics.mse_delta,
                    "psnr_delta": metrics.psnr_delta,
                },
                extra={
                    "dataset": "eurosat",
                    "model": "ConditionalUNet",
                    "base_channels": 16,
                    "time_dim": 64,
                    "scheduler": scheduler.state_dict(),
                    "sampler": args.sampler,
                    "sample_steps": args.sample_steps,
                    "ema_decay": args.ema_decay if ema is not None else None,
                    "image_size": 64,
                    "seed": args.seed,
                    "root": args.root,
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
            filename=f"ddpm_eurosat_epoch_{epoch:02d}.png",
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
            f"sample={sample_path.relative_to(PROJECT_ROOT)}",
            flush=True,
        )

    print(f"Best checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}", flush=True)


def _build_eval_model(model, ema, device):
    eval_model = copy.deepcopy(model).to(device)
    if ema is not None:
        ema.copy_to(eval_model)
    eval_model.eval()
    return eval_model


if __name__ == "__main__":
    main()
