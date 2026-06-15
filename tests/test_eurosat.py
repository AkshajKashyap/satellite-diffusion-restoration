import numpy as np
import torch
from PIL import Image

from satellite_diffusion_restoration.data import CorruptionConfig
from satellite_diffusion_restoration.data.eurosat import (
    EuroSATRestorationDataset,
    deterministic_split_indices,
)
from satellite_diffusion_restoration.training import build_restoration_datasets


class FakeImageDataset:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        value = index * 20
        array = np.full((20, 24, 3), value, dtype=np.uint8)
        return Image.fromarray(array), index


def test_eurosat_module_imports():
    import satellite_diffusion_restoration.data.eurosat as eurosat

    assert eurosat.EuroSATRestorationDataset is EuroSATRestorationDataset


def test_deterministic_split_indices_are_stable():
    first_train, first_val = deterministic_split_indices(
        20,
        val_fraction=0.25,
        seed=7,
        max_train_samples=5,
        max_val_samples=3,
    )
    second_train, second_val = deterministic_split_indices(
        20,
        val_fraction=0.25,
        seed=7,
        max_train_samples=5,
        max_val_samples=3,
    )
    other_train, other_val = deterministic_split_indices(
        20,
        val_fraction=0.25,
        seed=8,
        max_train_samples=5,
        max_val_samples=3,
    )

    assert first_train == second_train
    assert first_val == second_val
    assert (first_train, first_val) != (other_train, other_val)
    assert len(first_train) == 5
    assert len(first_val) == 3
    assert set(first_train).isdisjoint(first_val)


def test_eurosat_wrapper_with_fake_dataset_returns_pairs():
    config = CorruptionConfig(
        noise_std=0.0,
        mask_probability=0.0,
        cloud_probability=0.0,
        blur_probability=0.0,
    )
    dataset = EuroSATRestorationDataset(
        image_size=16,
        seed=3,
        corruption_config=config,
        indices=[2, 4],
        base_dataset=FakeImageDataset(),
    )

    corrupted, clean = dataset[0]

    assert len(dataset) == 2
    assert corrupted.shape == (3, 16, 16)
    assert clean.shape == (3, 16, 16)
    assert corrupted.dtype == torch.float32
    assert clean.dtype == torch.float32
    assert torch.allclose(corrupted, clean)
    assert float(clean.min()) >= 0.0
    assert float(clean.max()) <= 1.0


def test_dataset_factory_can_build_synthetic_datasets():
    train_dataset, val_dataset = build_restoration_datasets(
        dataset_name="synthetic",
        image_size=16,
        seed=5,
        max_train_samples=4,
        max_val_samples=2,
    )

    assert len(train_dataset) == 4
    assert len(val_dataset) == 2
