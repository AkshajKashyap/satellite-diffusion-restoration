import torch

from satellite_diffusion_restoration.data.synthetic import SyntheticSatelliteDataset


def test_synthetic_dataset_returns_corrupted_clean_pair():
    dataset = SyntheticSatelliteDataset(num_samples=2, image_size=32, seed=7)

    corrupted, clean = dataset[0]

    assert corrupted.shape == (3, 32, 32)
    assert clean.shape == (3, 32, 32)
    assert corrupted.dtype == torch.float32
    assert clean.dtype == torch.float32
    assert float(corrupted.min()) >= 0.0
    assert float(corrupted.max()) <= 1.0
    assert float(clean.min()) >= 0.0
    assert float(clean.max()) <= 1.0
