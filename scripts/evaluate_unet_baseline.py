"""Evaluate a saved U-Net baseline checkpoint on fresh synthetic data."""

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

    from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import UNet
    from satellite_diffusion_restoration.training import evaluate, get_device, load_checkpoint
    from satellite_diffusion_restoration.utils import seed_everything

    seed = 2024
    seed_everything(seed)
    device = get_device(prefer_cuda=False)

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "unet_baseline.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Run scripts/train_unet_baseline.py first."
        )

    model = UNet(base_channels=16).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device)

    dataset = SyntheticSatelliteDataset(
        num_samples=32,
        image_size=64,
        seed=seed,
        corruption_config=CorruptionConfig(),
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    metrics = evaluate(model, dataloader, device, nn.MSELoss())

    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    saved_paths = []
    model.eval()
    for index in range(4):
        corrupted, clean = dataset[index]
        with torch.no_grad():
            restored = model(corrupted.unsqueeze(0).to(device)).squeeze(0).cpu()
        saved_paths.append(
            save_comparison_grid(
                clean=clean,
                corrupted=corrupted,
                restored=restored,
                output_dir=sample_dir,
                filename=f"unet_baseline_eval_{index:02d}.png",
            )
        )

    ssim_text = "n/a" if metrics.ssim is None else f"{metrics.ssim:.4f}"
    print("U-Net baseline evaluation")
    print(f"Checkpoint: {checkpoint_path.relative_to(PROJECT_ROOT)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"PSNR: {metrics.psnr:.2f} dB")
    print(f"SSIM: {ssim_text}")
    print("Saved comparison grids:")
    for path in saved_paths:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
