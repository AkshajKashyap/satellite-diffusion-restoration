# Project Summary

## What Was Built

This repository implements a small but complete satellite image restoration benchmark under
synthetic corruptions. It includes synthetic data generation, EuroSAT RGB loading, seedable
corruptions, MSE/PSNR/SSIM metrics, visual grids, a residual U-Net baseline, an experimental
conditional DDPM, reports, a FastAPI endpoint, and a Streamlit demo.

## Why The Baseline Matters

The corrupted input is already a strong baseline because the synthetic corruptions often
leave much of the image intact. Every evaluation therefore reports corrupted-input metrics
beside model metrics. A model is not useful unless it beats returning the input.

## Why Residual Learning Helped

Directly predicting a clean image forced the U-Net to relearn the identity mapping. Residual
learning starts from the corrupted image and predicts a correction, which made the U-Net
immediately competitive and stable.

## How EuroSAT Was Used

EuroSAT provides real RGB satellite image patches as clean targets. The project applies
synthetic noise, masks, blur, and cloud-like overlays to create supervised restoration
pairs. This tests restoration on real image statistics while keeping corruptions controlled.

## Why DDPM Underperformed

The DDPM learned the noise-prediction objective and overfit a tiny fixed batch, so the
training path is not dead. It also performs well on low-noise one-step x0 diagnostics.
However, full sampling starts from random noise and requires a strong denoising trajectory
across many timesteps. With the current small model and training budget, sampled images are
worse than the corrupted input and far worse than the U-Net.

## Deployment Layer

The deployable path uses the residual U-Net. The FastAPI app exposes health and restoration
endpoints, and the Streamlit demo lets a user upload an image and run U-Net restoration.
DDPM is clearly labeled experimental and is not served by default.

## What I Would Improve Next

- Train the U-Net on larger and more varied corruption settings.
- Add real cloud or cloud-mask datasets if the goal becomes cloud removal.
- Add richer validation reports and per-corruption metrics.
- Investigate DDPM sampling from corrupted image plus noise instead of pure noise.
- Add EMA/schedule/capacity sweeps for DDPM on GPU.
- Package the API/demo with lightweight deployment instructions.
