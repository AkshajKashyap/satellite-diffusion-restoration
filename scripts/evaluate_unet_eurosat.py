"""Evaluate a saved EuroSAT residual U-Net checkpoint."""

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
    parser.add_argument("--root", default="data/raw", help="EuroSAT root directory.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unet_eurosat.pt")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=2024)
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
        build_restoration_datasets,
        evaluate,
        load_checkpoint,
    )
    from satellite_diffusion_restoration.utils import seed_everything

    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Missing checkpoint: {checkpoint_path}. Run scripts/train_unet_eurosat.py first."
        )

    checkpoint_preview = torch.load(checkpoint_path, map_location=device)
    extra = checkpoint_preview.get("extra", {})
    model = UNet(
        base_channels=extra.get("base_channels", 16),
        residual_mode=extra.get("residual_mode", True),
        residual_scale=extra.get("residual_scale", 0.5),
    ).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device)

    try:
        _, dataset = build_restoration_datasets(
            dataset_name="eurosat",
            root=args.root,
            download=False,
            image_size=extra.get("image_size", 64),
            seed=args.seed,
            max_train_samples=1,
            max_val_samples=args.max_samples,
            corruption_config=CorruptionConfig(),
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nDownload EuroSAT first:\n"
            "  python scripts/train_unet_eurosat.py --download"
        ) from exc

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(model, dataloader, device, nn.MSELoss())

    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    saved_paths = []
    model.eval()
    for index in range(min(4, len(dataset))):
        corrupted, clean = dataset[index]
        with torch.no_grad():
            restored = model(corrupted.unsqueeze(0).to(device)).squeeze(0).cpu()
        saved_paths.append(
            save_comparison_grid(
                clean=clean,
                corrupted=corrupted,
                restored=restored,
                output_dir=sample_dir,
                filename=f"unet_eurosat_eval_{index:02d}.png",
            )
        )

    corrupted_ssim = "n/a" if metrics.corrupted_ssim is None else f"{metrics.corrupted_ssim:.4f}"
    restored_ssim = "n/a" if metrics.restored_ssim is None else f"{metrics.restored_ssim:.4f}"
    print("EuroSAT residual U-Net evaluation")
    print(f"Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Samples: {len(dataset)}")
    print(
        "Corrupted input vs clean: "
        f"MSE={metrics.corrupted_mse:.6f}, "
        f"PSNR={metrics.corrupted_psnr:.2f} dB, "
        f"SSIM={corrupted_ssim}"
    )
    print(
        "Restored model vs clean: "
        f"MSE={metrics.restored_mse:.6f}, "
        f"PSNR={metrics.restored_psnr:.2f} dB, "
        f"SSIM={restored_ssim}"
    )
    print(
        "Improvement: "
        f"MSE delta={metrics.mse_delta:+.6f}, "
        f"PSNR delta={metrics.psnr_delta:+.2f} dB"
    )
    print("Saved comparison grids:")
    for path in saved_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
