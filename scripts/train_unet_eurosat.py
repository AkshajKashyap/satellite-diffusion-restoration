"""Train the residual U-Net baseline on EuroSAT RGB images with synthetic corruptions."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    from torch import nn
    from torch.utils.data import DataLoader

    from satellite_diffusion_restoration.data import CorruptionConfig
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import UNet
    from satellite_diffusion_restoration.training import (
        History,
        build_restoration_datasets,
        evaluate,
        save_checkpoint,
        train_one_epoch,
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
            "  python scripts/train_unet_eurosat.py --download"
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    base_channels = 16
    residual_scale = 0.5
    model = UNet(
        base_channels=base_channels,
        residual_mode=True,
        residual_scale=residual_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    history = History()
    best_val_loss = float("inf")
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "unet_eurosat.pt"
    sample_dir = PROJECT_ROOT / "outputs" / "samples"

    print("EuroSAT residual U-Net training")
    print(f"Device: {device}")
    print(
        "Config: "
        f"root={args.root}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = evaluate(model, val_loader, device, loss_fn)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_metrics.loss)
        history.val_psnr.append(val_metrics.restored_psnr)
        if val_metrics.restored_ssim is not None:
            history.val_ssim.append(val_metrics.restored_ssim)

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
                    "corrupted_mse": val_metrics.corrupted_mse,
                    "corrupted_psnr": val_metrics.corrupted_psnr,
                    "corrupted_ssim": (
                        val_metrics.corrupted_ssim
                        if val_metrics.corrupted_ssim is not None
                        else float("nan")
                    ),
                    "restored_mse": val_metrics.restored_mse,
                    "restored_psnr": val_metrics.restored_psnr,
                    "restored_ssim": (
                        val_metrics.restored_ssim
                        if val_metrics.restored_ssim is not None
                        else float("nan")
                    ),
                    "mse_delta": val_metrics.mse_delta,
                    "psnr_delta": val_metrics.psnr_delta,
                },
                extra={
                    "dataset": "eurosat",
                    "model": "UNet",
                    "base_channels": base_channels,
                    "residual_mode": True,
                    "residual_scale": residual_scale,
                    "image_size": 64,
                    "seed": args.seed,
                    "root": args.root,
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
            filename=f"unet_eurosat_epoch_{epoch:02d}.png",
        )

        corrupted_ssim = (
            "n/a"
            if val_metrics.corrupted_ssim is None
            else f"{val_metrics.corrupted_ssim:.4f}"
        )
        restored_ssim = (
            "n/a" if val_metrics.restored_ssim is None else f"{val_metrics.restored_ssim:.4f}"
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"corrupted_mse={val_metrics.corrupted_mse:.6f} | "
            f"restored_mse={val_metrics.restored_mse:.6f} | "
            f"mse_delta={val_metrics.mse_delta:+.6f} | "
            f"corrupted_psnr={val_metrics.corrupted_psnr:.2f} dB | "
            f"restored_psnr={val_metrics.restored_psnr:.2f} dB | "
            f"psnr_delta={val_metrics.psnr_delta:+.2f} dB | "
            f"corrupted_ssim={corrupted_ssim} | restored_ssim={restored_ssim} | "
            f"sample={sample_path.relative_to(PROJECT_ROOT)}"
        )

    print(f"Best checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
