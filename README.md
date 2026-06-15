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
image tensor and predicts the clean RGB image tensor using MSE loss. The default training
run is intentionally tiny and CPU-compatible, so it is useful as a pipeline smoke test,
not as a real benchmark.

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
- train samples: `128`
- validation samples: `32`
- batch size: `16`
- epochs: `3`
- learning rate: `1e-3`
- device: CPU by default

Expected outputs:

- best checkpoint: `outputs/checkpoints/unet_baseline.pt`
- epoch sample grids: `outputs/samples/unet_baseline_epoch_01.png`,
  `outputs/samples/unet_baseline_epoch_02.png`, and
  `outputs/samples/unet_baseline_epoch_03.png`

## Evaluate U-Net Baseline

```bash
python scripts/evaluate_unet_baseline.py
```

Expected outputs:

- final MSE, PSNR, and SSIM printed in the terminal
- evaluation sample grids in `outputs/samples/`, such as
  `outputs/samples/unet_baseline_eval_00.png`

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
