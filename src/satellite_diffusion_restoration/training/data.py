"""Dataset builders used by training and evaluation scripts."""

from __future__ import annotations

from typing import Any

from satellite_diffusion_restoration.data import CorruptionConfig, SyntheticSatelliteDataset


def build_synthetic_train_val_datasets(
    train_samples: int = 128,
    val_samples: int = 32,
    image_size: int = 64,
    seed: int = 42,
    corruption_config: CorruptionConfig | dict[str, Any] | None = None,
) -> tuple[SyntheticSatelliteDataset, SyntheticSatelliteDataset]:
    """Build train and validation synthetic datasets with different sample seeds."""
    train_dataset = SyntheticSatelliteDataset(
        num_samples=train_samples,
        image_size=image_size,
        seed=seed,
        corruption_config=corruption_config,
    )
    val_dataset = SyntheticSatelliteDataset(
        num_samples=val_samples,
        image_size=image_size,
        seed=seed + 50_000,
        corruption_config=corruption_config,
    )
    return train_dataset, val_dataset
