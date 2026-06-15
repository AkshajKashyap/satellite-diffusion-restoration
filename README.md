# Satellite Diffusion Restoration

U-Net and DDPM satellite image restoration project for synthetic cloud, noise, and blur
removal. The intended long-term target is EuroSAT-style RGB imagery, but the current
milestones are fully runnable without downloaded data.

## Current Milestones

### 1. Data and Corruption Foundation

The repository includes:

- seedable tensor corruptions for Gaussian noise, rectangular masks, soft cloud-like blobs,
  and Gaussian blur
- a synthetic pseudo-satellite dataset that returns `(corrupted, clean)` tensor pairs
- baseline MSE, PSNR, and SSIM metrics
- comparison-grid visualization helpers
- a smoke script that exercises the data, corruption, metrics, and visualization path

### 2. U-Net Baseline

The repository now includes a small supervised U-Net denoiser. It takes a corrupted RGB
image tensor and predicts a cleaned RGB image tensor using MSE loss. The default model uses
residual restoration:

```text
restored = clamp(corrupted + predicted_residual, 0, 1)
```

Residual mode starts as an identity mapping, so the model begins at the corrupted-input
baseline and learns corrections from there. Direct clean-image prediction is still
available in the model, but residual mode is the default because the corrupted image is
already a strong baseline.

Every evaluation reports both corrupted-input metrics and restored-model metrics. This is
important: a denoiser is not useful unless it beats simply returning the corrupted input.
In the current synthetic setup, a fresh evaluation run after baseline training produced:

- corrupted input: MSE `0.027949`, PSNR `15.54 dB`, SSIM `0.4307`
- restored model: MSE `0.007120`, PSNR `21.48 dB`, SSIM `0.4993`
- improvement: MSE delta `+0.020829`, PSNR delta `+5.94 dB`

These are smoke-test results on generated data, not a real satellite benchmark.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Smoke Run

```bash
python scripts/smoke_data_pipeline.py
```

Expected output:

- a short terminal summary with MSE, PSNR, and SSIM for corrupted images against clean images
- sample comparison grids in `outputs/samples/`, such as
  `outputs/samples/synthetic_sample_00.png`

## Train U-Net Baseline

```bash
python scripts/train_unet_baseline.py
```

Default training setup:

- image size: `64`
- train samples: `256`
- validation samples: `64`
- batch size: `16`
- epochs: `8`
- learning rate: `1e-3`
- device: CPU by default
- restoration mode: residual U-Net

Expected outputs:

- best checkpoint: `outputs/checkpoints/unet_baseline.pt`
- epoch sample grids: `outputs/samples/unet_baseline_epoch_01.png`,
  `outputs/samples/unet_baseline_epoch_02.png`, and later epoch files

## Evaluate U-Net Baseline

```bash
python scripts/evaluate_unet_baseline.py
```

Expected outputs:

- final MSE, PSNR, and SSIM printed in the terminal
- corrupted-input baseline metrics printed alongside restored-model metrics
- MSE and PSNR improvement deltas
- evaluation sample grids in `outputs/samples/`, such as
  `outputs/samples/unet_baseline_eval_00.png`

## Tiny Overfit Sanity Check

```bash
python scripts/overfit_unet_tiny.py
```

This debugging script trains on one tiny fixed synthetic batch. It should drop the training
loss meaningfully and save:

- `outputs/samples/unet_tiny_overfit_before.png`
- `outputs/samples/unet_tiny_overfit_after.png`

If this script cannot overfit, the model, loss, optimizer, or target/input wiring likely
has a bug.

## Quality Checks

```bash
pytest -q
ruff check .
```

## Limitations

- training and evaluation are synthetic-only
- the default U-Net run is tiny CPU training, not a real satellite benchmark
- no real EuroSAT loader or downloader yet
- no DDPM restoration or noise-prediction model yet
- no FastAPI service yet
- no Streamlit demo yet
- no GPU-specific workflow is required or tuned yet
