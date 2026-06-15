"""Run a tiny synthetic data, corruption, metrics, and visualization smoke test."""

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

    from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
    from satellite_diffusion_restoration.evaluation import mse, psnr, save_comparison_grid, ssim_batch
    from satellite_diffusion_restoration.utils import seed_everything

    seed_everything(42)

    dataset = SyntheticSatelliteDataset(
        num_samples=8,
        image_size=64,
        seed=42,
        corruption_config=CorruptionConfig(
            noise_std=0.06,
            mask_probability=0.65,
            cloud_probability=0.9,
            blur_probability=0.35,
        ),
    )

    output_dir = PROJECT_ROOT / "outputs" / "samples"
    clean_images = []
    corrupted_images = []
    saved_paths = []

    for index in range(len(dataset)):
        corrupted, clean = dataset[index]
        clean_images.append(clean)
        corrupted_images.append(corrupted)

        if index < 4:
            saved_paths.append(
                save_comparison_grid(
                    clean=clean,
                    corrupted=corrupted,
                    output_dir=output_dir,
                    filename=f"synthetic_sample_{index:02d}.png",
                )
            )

    clean_batch = torch.stack(clean_images)
    corrupted_batch = torch.stack(corrupted_images)

    print("Synthetic data pipeline smoke run")
    print(f"Samples: {len(dataset)}")
    print(f"Image shape: {tuple(clean_batch.shape[1:])}")
    print(f"MSE corrupted vs clean: {mse(corrupted_batch, clean_batch):.6f}")
    print(f"PSNR corrupted vs clean: {psnr(corrupted_batch, clean_batch):.2f} dB")
    print(f"SSIM corrupted vs clean: {ssim_batch(corrupted_batch, clean_batch):.4f}")
    print("Saved comparison grids:")
    for path in saved_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
