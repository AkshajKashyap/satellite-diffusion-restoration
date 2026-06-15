"""Dataset builders used by training and evaluation scripts."""

from __future__ import annotations

from typing import Any

from satellite_diffusion_restoration.data import (
    CorruptionConfig,
    EuroSATRestorationDataset,
    SyntheticSatelliteDataset,
    build_eurosat_train_val_datasets,
)


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


def build_restoration_datasets(
    dataset_name: str = "synthetic",
    train_samples: int = 256,
    val_samples: int = 64,
    image_size: int = 64,
    seed: int = 42,
    corruption_config: CorruptionConfig | dict[str, Any] | None = None,
    root: str = "data/raw",
    download: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    val_fraction: float = 0.2,
) -> tuple[SyntheticSatelliteDataset, SyntheticSatelliteDataset] | tuple[
    EuroSATRestorationDataset, EuroSATRestorationDataset
]:
    """Build train/validation restoration datasets by name."""
    normalized_name = dataset_name.lower()
    train_count = max_train_samples if max_train_samples is not None else train_samples
    val_count = max_val_samples if max_val_samples is not None else val_samples

    if normalized_name == "synthetic":
        return build_synthetic_train_val_datasets(
            train_samples=train_count,
            val_samples=val_count,
            image_size=image_size,
            seed=seed,
            corruption_config=corruption_config,
        )

    if normalized_name == "eurosat":
        return build_eurosat_train_val_datasets(
            root=root,
            image_size=image_size,
            download=download,
            seed=seed,
            max_train_samples=train_count,
            max_val_samples=val_count,
            val_fraction=val_fraction,
            corruption_config=corruption_config,
        )

    raise ValueError(f"Unknown dataset_name: {dataset_name!r}. Expected 'synthetic' or 'eurosat'.")
