import torch

from satellite_diffusion_restoration.evaluation.metrics import psnr


def test_psnr_is_higher_for_identical_images_than_corrupted_images():
    clean = torch.full((3, 16, 16), 0.5)
    corrupted = (clean + 0.1).clamp(0.0, 1.0)

    assert psnr(clean, clean) > psnr(corrupted, clean)
