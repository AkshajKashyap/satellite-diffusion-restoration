import torch

from satellite_diffusion_restoration.data.corruptions import CombinedCorruption, CorruptionConfig


def test_corruption_output_shape_and_range():
    image = torch.full((3, 32, 32), 0.5)
    generator = torch.Generator().manual_seed(123)
    transform = CombinedCorruption(CorruptionConfig())

    corrupted = transform(image, generator=generator)

    assert corrupted.shape == image.shape
    assert float(corrupted.min()) >= 0.0
    assert float(corrupted.max()) <= 1.0
