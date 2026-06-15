"""Evaluate a conditional DDPM restoration checkpoint on EuroSAT."""

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
    parser.add_argument("--checkpoint", default="outputs/checkpoints/ddpm_eurosat.pt")
    parser.add_argument("--unet-checkpoint", default="outputs/checkpoints/unet_eurosat.pt")
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
    from satellite_diffusion_restoration.models import ConditionalUNet, DDPMScheduler, UNet
    from satellite_diffusion_restoration.training import (
        build_restoration_datasets,
        evaluate,
        evaluate_diffusion_restoration,
        load_checkpoint,
        sample_diffusion_restoration,
    )
    from satellite_diffusion_restoration.utils import seed_everything

    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Missing checkpoint: {checkpoint_path}. Run scripts/train_ddpm_eurosat.py first."
        )

    checkpoint_preview = torch.load(checkpoint_path, map_location=device)
    extra = checkpoint_preview.get("extra", {})
    scheduler_config = extra.get("scheduler", {"timesteps": 100})
    scheduler = DDPMScheduler(**scheduler_config).to(device)
    model = ConditionalUNet(
        base_channels=extra.get("base_channels", 16),
        time_dim=extra.get("time_dim", 64),
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
            "  python scripts/train_ddpm_eurosat.py --download"
        ) from exc

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    ddpm_metrics = evaluate_diffusion_restoration(model, scheduler, dataloader, device)

    unet_metrics = None
    unet_checkpoint_path = PROJECT_ROOT / args.unet_checkpoint
    if unet_checkpoint_path.exists():
        unet_preview = torch.load(unet_checkpoint_path, map_location=device)
        unet_extra = unet_preview.get("extra", {})
        unet_model = UNet(
            base_channels=unet_extra.get("base_channels", 16),
            residual_mode=unet_extra.get("residual_mode", True),
            residual_scale=unet_extra.get("residual_scale", 0.5),
        ).to(device)
        load_checkpoint(unet_checkpoint_path, model=unet_model, device=device)
        unet_metrics = evaluate(unet_model, dataloader, device, nn.MSELoss())

    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    saved_paths = []
    model.eval()
    for index in range(min(4, len(dataset))):
        corrupted, clean = dataset[index]
        restored = sample_diffusion_restoration(
            model,
            scheduler,
            corrupted.unsqueeze(0),
            device,
        ).squeeze(0)
        saved_paths.append(
            save_comparison_grid(
                clean=clean,
                corrupted=corrupted,
                restored=restored,
                output_dir=sample_dir,
                filename=f"ddpm_eurosat_eval_{index:02d}.png",
            )
        )

    print("EuroSAT conditional DDPM evaluation")
    print(f"Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Samples: {len(dataset)}")
    _print_metric_block("Corrupted input", ddpm_metrics.corrupted_mse, ddpm_metrics.corrupted_psnr, ddpm_metrics.corrupted_ssim)
    _print_metric_block("DDPM restored", ddpm_metrics.restored_mse, ddpm_metrics.restored_psnr, ddpm_metrics.restored_ssim)
    print(
        "DDPM improvement vs corrupted: "
        f"MSE delta={ddpm_metrics.mse_delta:+.6f}, "
        f"PSNR delta={ddpm_metrics.psnr_delta:+.2f} dB"
    )

    if unet_metrics is not None:
        _print_metric_block(
            "U-Net restored",
            unet_metrics.restored_mse,
            unet_metrics.restored_psnr,
            unet_metrics.restored_ssim,
        )
        print(
            "DDPM vs U-Net: "
            f"MSE delta={unet_metrics.restored_mse - ddpm_metrics.restored_mse:+.6f}, "
            f"PSNR delta={ddpm_metrics.restored_psnr - unet_metrics.restored_psnr:+.2f} dB"
        )
    else:
        print(f"U-Net checkpoint not found at {unet_checkpoint_path}; skipped U-Net comparison.")

    print("Saved comparison grids:")
    for path in saved_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


def _print_metric_block(label: str, mse_value: float, psnr_value: float, ssim_value: float | None) -> None:
    ssim_text = "n/a" if ssim_value is None else f"{ssim_value:.4f}"
    print(f"{label}: MSE={mse_value:.6f}, PSNR={psnr_value:.2f} dB, SSIM={ssim_text}")


if __name__ == "__main__":
    main()
