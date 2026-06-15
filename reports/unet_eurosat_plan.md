# EuroSAT U-Net Benchmark Plan

## Why Real-Data Evaluation Matters

Synthetic pseudo-satellite images are useful for checking tensor shapes, corruption logic,
metrics, checkpointing, and training stability. They do not prove that a restoration model
has learned useful structure for real satellite imagery. EuroSAT adds real RGB land-cover
textures, class variation, edges, roads, vegetation, water, and sensor-like image statistics
that the synthetic generator only approximates.

## What Synthetic Corruption Proves

The corruption pipeline can test whether a supervised restoration model learns to remove
the exact synthetic corruptions used in training:

- Gaussian noise
- rectangular missing regions
- soft cloud-like overlays
- optional blur

This proves the pipeline is coherent and the U-Net can beat returning the corrupted input.
It does not prove robustness to real clouds, real atmospheric effects, sensor artifacts, or
out-of-distribution regions.

## Expected Limitations

- EuroSAT images are already 64x64 RGB patches, not full-scene satellite products.
- Corruptions are synthetic, even when the clean images are real.
- The benchmark uses a deterministic split because torchvision EuroSAT has no official
  train/validation split.
- Default training is intentionally small enough for portfolio and CPU smoke runs.
- Metrics are full-image MSE, PSNR, and SSIM; they do not measure semantic usefulness.

## Metrics To Report

Every run should report both baselines:

- corrupted-input MSE, PSNR, and SSIM
- restored-model MSE, PSNR, and SSIM
- MSE delta: corrupted MSE minus restored MSE
- PSNR delta: restored PSNR minus corrupted PSNR

The model should not be considered useful unless it beats the corrupted-input baseline on
MSE and PSNR.

## Good Enough Before DDPM

Before moving to DDPM, the residual U-Net should:

- reliably overfit a tiny EuroSAT subset
- beat corrupted-input MSE and PSNR on a held-out deterministic EuroSAT subset
- produce visual samples that remove obvious synthetic corruption without destroying image
  structure
- keep the benchmark reproducible from one command and one checkpoint path

DDPM work should wait until this real-data supervised baseline is stable and honestly
reported.
