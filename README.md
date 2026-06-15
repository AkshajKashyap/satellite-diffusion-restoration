# Satellite Diffusion Restoration

U-Net and DDPM satellite image restoration project for synthetic cloud, noise, and blur
removal. The intended long-term target is EuroSAT-style RGB imagery, but the first
milestone is fully runnable without downloaded data.

## First Milestone

This repository currently provides a clean foundation for future restoration training:

- seedable tensor corruptions for Gaussian noise, rectangular masks, soft cloud-like blobs,
  and Gaussian blur
- a synthetic pseudo-satellite dataset that returns `(corrupted, clean)` tensor pairs
- baseline MSE, PSNR, and SSIM metrics
- comparison-grid visualization helpers
- a smoke script that exercises the data, corruption, metrics, and visualization path

It does not train a model yet. The goal is to make the data path testable before adding a
U-Net or diffusion model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Run

```bash
python scripts/smoke_data_pipeline.py
```

Expected output:

- a short terminal summary with MSE, PSNR, and SSIM for corrupted images against clean images
- sample comparison grids in `outputs/samples/`, such as
  `outputs/samples/synthetic_sample_00.png`

## Quality Checks

```bash
pytest -q
ruff check .
```

## Not Implemented Yet

- U-Net restoration model
- DDPM restoration or noise-prediction model
- training loops
- FastAPI service
- Streamlit demo
- real EuroSAT loader
- GPU-specific workflows
