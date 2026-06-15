"""Diagnose one-step and full-sampling DDPM restoration behavior."""

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
    parser.add_argument("--checkpoint", default="outputs/checkpoints/ddpm_synthetic.pt")
    parser.add_argument("--dataset", choices=["synthetic", "eurosat"], default="synthetic")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--diagnostic-timesteps", default="10,50,90")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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

    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import ConditionalUNet, DDPMScheduler
    from satellite_diffusion_restoration.training import (
        compute_diffusion_diagnostics,
        EMAModel,
        load_checkpoint,
        one_step_x0_diagnostic,
        sample_diffusion_restoration,
    )
    from satellite_diffusion_restoration.utils import seed_everything

    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint = None
    extra = {}
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        extra = checkpoint.get("extra", {})
    else:
        print(f"Checkpoint not found at {checkpoint_path}; using an untrained diagnostic model.")

    scheduler_config = extra.get("scheduler", {"timesteps": args.timesteps})
    scheduler = DDPMScheduler(**scheduler_config).to(device)
    model = ConditionalUNet(
        base_channels=extra.get("base_channels", 16),
        time_dim=extra.get("time_dim", 64),
    ).to(device)

    using_ema = False
    if checkpoint is not None:
        load_checkpoint(checkpoint_path, model=model, device=device)
        if not args.disable_ema and "ema_state_dict" in checkpoint:
            ema = EMAModel(model)
            ema.load_state_dict(checkpoint["ema_state_dict"], device=device)
            ema.copy_to(model)
            using_ema = True

    corrupted, clean = _load_sample(args, extra)
    corrupted_batch = corrupted.unsqueeze(0)
    clean_batch = clean.unsqueeze(0)
    diagnostic_timesteps = _parse_timesteps(args.diagnostic_timesteps)
    diagnostics = compute_diffusion_diagnostics(
        model,
        scheduler,
        corrupted_batch,
        clean_batch,
        device,
        timesteps=diagnostic_timesteps,
        sampler=args.sampler,
        sample_steps=args.sample_steps,
        compute_ssim=True,
    )

    first_timestep = min(max(diagnostic_timesteps[0], 0), scheduler.timesteps - 1)
    one_step_image, _ = one_step_x0_diagnostic(
        model,
        scheduler,
        corrupted_batch,
        clean_batch,
        timestep=first_timestep,
        device=device,
    )
    sampled_image = sample_diffusion_restoration(
        model,
        scheduler,
        corrupted_batch,
        device,
        sampler=args.sampler,
        sample_steps=args.sample_steps,
    )

    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    one_step_path = save_comparison_grid(
        clean=clean,
        corrupted=corrupted,
        restored=one_step_image[0],
        output_dir=sample_dir,
        filename=f"ddpm_diagnostic_one_step_t{first_timestep}.png",
    )
    sampled_path = save_comparison_grid(
        clean=clean,
        corrupted=corrupted,
        restored=sampled_image[0],
        output_dir=sample_dir,
        filename="ddpm_diagnostic_sampled.png",
    )

    print("DDPM reconstruction diagnostic")
    print(f"Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT) if checkpoint_path.exists() else 'untrained'}")
    print(f"Dataset: {args.dataset}")
    print(f"Using EMA weights: {using_ema}")
    print(f"Sampler: {args.sampler}, sample_steps={args.sample_steps}")
    _print_metric_dict("Corrupted input", diagnostics["corrupted"])
    for timestep, metrics in diagnostics["one_step"].items():
        _print_metric_dict(f"One-step x0 t={timestep}", metrics)
    _print_metric_dict("Full sampled DDPM", diagnostics["sampled"])
    print("Saved diagnostic grids:")
    print(f"  - {one_step_path.relative_to(PROJECT_ROOT)}")
    print(f"  - {sampled_path.relative_to(PROJECT_ROOT)}")


def _load_sample(args: argparse.Namespace, extra: dict):
    from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
    from satellite_diffusion_restoration.training import build_restoration_datasets

    if args.dataset == "synthetic":
        dataset = SyntheticSatelliteDataset(
            num_samples=1,
            image_size=extra.get("image_size", 64),
            seed=args.seed,
            corruption_config=CorruptionConfig(),
        )
        return dataset[0]

    try:
        _, dataset = build_restoration_datasets(
            dataset_name="eurosat",
            root=args.root,
            download=False,
            image_size=extra.get("image_size", 64),
            seed=args.seed,
            max_train_samples=1,
            max_val_samples=1,
            corruption_config=CorruptionConfig(),
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nDownload EuroSAT first:\n"
            "  python scripts/train_ddpm_eurosat.py --download"
        ) from exc
    return dataset[0]


def _parse_timesteps(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _print_metric_dict(label: str, metrics: object) -> None:
    if not isinstance(metrics, dict):
        raise TypeError("Diagnostic metrics should be dictionaries")
    ssim_value = metrics.get("ssim")
    ssim_text = "n/a" if ssim_value is None else f"{float(ssim_value):.4f}"
    print(
        f"{label}: MSE={float(metrics['mse']):.6f}, "
        f"PSNR={float(metrics['psnr']):.2f} dB, SSIM={ssim_text}"
    )


if __name__ == "__main__":
    main()
