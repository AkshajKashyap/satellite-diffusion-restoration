"""Overfit a tiny synthetic batch to sanity-check the U-Net training path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/satellite_diffusion_restoration_matplotlib")


def run_tiny_overfit(
    num_samples: int = 8,
    image_size: int = 32,
    steps: int = 160,
    learning_rate: float = 2e-3,
    base_channels: int = 8,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Train on one tiny fixed batch and return the starting and ending loss."""
    import torch
    from torch import nn

    from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import UNet
    from satellite_diffusion_restoration.utils import seed_everything

    seed_everything(123)
    dataset = SyntheticSatelliteDataset(
        num_samples=num_samples,
        image_size=image_size,
        seed=123,
        corruption_config=CorruptionConfig(),
    )
    corrupted_batch = torch.stack([dataset[index][0] for index in range(num_samples)])
    clean_batch = torch.stack([dataset[index][1] for index in range(num_samples)])

    model = UNet(
        base_channels=base_channels,
        residual_mode=True,
        residual_scale=0.5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        before_restored = model(corrupted_batch)
        starting_loss = float(loss_fn(before_restored, clean_batch).item())

    saved_paths: list[Path] = []
    sample_dir = PROJECT_ROOT / "outputs" / "samples"
    if save_outputs:
        saved_paths.append(
            save_comparison_grid(
                clean=clean_batch[0],
                corrupted=corrupted_batch[0],
                restored=before_restored[0],
                output_dir=sample_dir,
                filename="unet_tiny_overfit_before.png",
            )
        )

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        restored = model(corrupted_batch)
        loss = loss_fn(restored, clean_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        after_restored = model(corrupted_batch)
        ending_loss = float(loss_fn(after_restored, clean_batch).item())

    if save_outputs:
        saved_paths.append(
            save_comparison_grid(
                clean=clean_batch[0],
                corrupted=corrupted_batch[0],
                restored=after_restored[0],
                output_dir=sample_dir,
                filename="unet_tiny_overfit_after.png",
            )
        )

    return {
        "starting_loss": starting_loss,
        "ending_loss": ending_loss,
        "loss_delta": starting_loss - ending_loss,
        "saved_paths": saved_paths,
    }


def main() -> None:
    result = run_tiny_overfit()
    starting_loss = float(result["starting_loss"])
    ending_loss = float(result["ending_loss"])
    loss_delta = float(result["loss_delta"])
    print("Tiny U-Net overfit sanity check")
    print(f"Starting train loss: {starting_loss:.6f}")
    print(f"Ending train loss: {ending_loss:.6f}")
    print(f"Loss delta: {loss_delta:+.6f}")
    print("Saved comparison grids:")
    for path in result["saved_paths"]:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
