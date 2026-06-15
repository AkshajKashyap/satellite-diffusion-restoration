"""Overfit a tiny conditional DDPM noise-prediction batch for debugging."""

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
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def run_tiny_ddpm_overfit(
    num_samples: int = 8,
    image_size: int = 32,
    steps: int = 300,
    learning_rate: float = 1e-3,
    timesteps: int = 50,
    sample_steps: int = 25,
    seed: int = 123,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Overfit one fixed noising batch and return loss plus sample paths."""
    import torch
    from torch import nn

    from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset
    from satellite_diffusion_restoration.evaluation import save_comparison_grid
    from satellite_diffusion_restoration.models import ConditionalUNet, DDPMScheduler
    from satellite_diffusion_restoration.training import sample_diffusion_restoration
    from satellite_diffusion_restoration.utils import seed_everything

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticSatelliteDataset(
        num_samples=num_samples,
        image_size=image_size,
        seed=seed,
        corruption_config=CorruptionConfig(),
    )
    corrupted = torch.stack([dataset[index][0] for index in range(num_samples)]).to(device)
    clean = torch.stack([dataset[index][1] for index in range(num_samples)]).to(device)

    model = ConditionalUNet(base_channels=8, time_dim=32).to(device)
    scheduler = DDPMScheduler(timesteps=timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    fixed_timesteps = torch.randint(0, timesteps, (num_samples,), device=device, dtype=torch.long)
    fixed_noise = torch.randn_like(clean)
    noisy_clean = scheduler.q_sample(clean, fixed_timesteps, fixed_noise)

    model.eval()
    with torch.no_grad():
        starting_loss = float(loss_fn(model(noisy_clean, corrupted, fixed_timesteps), fixed_noise).item())
        before_sample = sample_diffusion_restoration(
            model,
            scheduler,
            corrupted[:1],
            device,
            sampler="ddim",
            sample_steps=sample_steps,
        )[0]

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted_noise = model(noisy_clean, corrupted, fixed_timesteps)
        loss = loss_fn(predicted_noise, fixed_noise)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        ending_loss = float(loss_fn(model(noisy_clean, corrupted, fixed_timesteps), fixed_noise).item())
        after_sample = sample_diffusion_restoration(
            model,
            scheduler,
            corrupted[:1],
            device,
            sampler="ddim",
            sample_steps=sample_steps,
        )[0]

    saved_paths: list[Path] = []
    if save_outputs:
        sample_dir = PROJECT_ROOT / "outputs" / "samples"
        saved_paths.append(
            save_comparison_grid(
                clean=clean[0].cpu(),
                corrupted=corrupted[0].cpu(),
                restored=before_sample.cpu(),
                output_dir=sample_dir,
                filename="ddpm_tiny_overfit_before.png",
            )
        )
        saved_paths.append(
            save_comparison_grid(
                clean=clean[0].cpu(),
                corrupted=corrupted[0].cpu(),
                restored=after_sample.cpu(),
                output_dir=sample_dir,
                filename="ddpm_tiny_overfit_after.png",
            )
        )

    return {
        "starting_loss": starting_loss,
        "ending_loss": ending_loss,
        "loss_delta": starting_loss - ending_loss,
        "dropped_meaningfully": ending_loss < starting_loss * 0.5,
        "saved_paths": saved_paths,
    }


def main() -> None:
    args = parse_args()
    result = run_tiny_ddpm_overfit(
        num_samples=args.num_samples,
        image_size=args.image_size,
        steps=args.steps,
        learning_rate=args.lr,
        timesteps=args.timesteps,
        sample_steps=args.sample_steps,
        seed=args.seed,
    )
    print("Tiny DDPM overfit sanity check")
    print(f"Starting noise loss: {float(result['starting_loss']):.6f}")
    print(f"Ending noise loss: {float(result['ending_loss']):.6f}")
    print(f"Loss delta: {float(result['loss_delta']):+.6f}")
    print(f"Dropped meaningfully: {result['dropped_meaningfully']}")
    print("Saved comparison grids:")
    for path in result["saved_paths"]:
        print(f"  - {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
