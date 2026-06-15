# Interview Notes

## 30-Second Explanation

This project is a satellite image restoration benchmark under synthetic corruptions. It
uses synthetic data for smoke tests and EuroSAT RGB images for real-image evaluation. The
main working model is a residual U-Net that improves EuroSAT PSNR from `15.57 dB` for the
corrupted input to `23.70 dB`. I also implemented a conditional DDPM, diagnosed why it
underperforms, and shipped API and Streamlit demos around the reliable U-Net path.

## 2-Minute Technical Explanation

The repo builds clean `(corrupted, clean)` image pairs using Gaussian noise, masks,
cloud-like blobs, and blur. Every model is evaluated against the corrupted-input baseline
because simply returning the input is surprisingly strong. The residual U-Net predicts a
correction to the corrupted image, which made training stable and produced a strong
EuroSAT benchmark result. The DDPM is conditional: it receives noised clean image `x_t`,
timestep embedding, and corrupted image conditioning, then predicts diffusion noise. It can
overfit tiny noise-prediction data and performs well on low-noise one-step diagnostics, but
full sampling from noise fails, so it remains experimental.

## Problem Solved

The project creates a reproducible restoration benchmark and deployable U-Net baseline for
satellite image restoration under synthetic corruptions.

## Why Residual Learning Helped

The corrupted image is already close to the target. Residual learning starts from that
useful input and learns a correction instead of forcing the model to reproduce the entire
image from scratch.

## Why The Corrupted-Input Baseline Mattered

Without that baseline, a model can look useful while actually making images worse. Every
evaluation reports corrupted-input metrics beside model metrics.

## Why DDPM Underperformed

The DDPM learned low-noise denoising and a fixed tiny objective, but full restoration
requires a strong iterative denoising trajectory from random noise. The current small model
and training budget are not enough for that.

## What I Would Improve Next

- Add per-corruption and per-class EuroSAT metrics.
- Train the U-Net on broader corruption distributions.
- Try DDPM sampling from corrupted image plus noise.
- Tune DDPM beta schedules, EMA decay, timestep count, and model capacity on GPU.
- Add real cloud-mask data before making any cloud-removal claims.

## Likely Interview Questions

**Why not use DDPM as the main model?**  
Because full sampled DDPM restoration underperforms the corrupted input and U-Net. The
repo reports that honestly.

**What was the biggest debugging insight?**  
One-step DDPM reconstruction works at low noise, but full sampling fails. That isolates the
problem to high-noise trajectory/sampling quality rather than total implementation failure.

**What makes the project reproducible?**  
Seeded synthetic data, deterministic EuroSAT splits, saved checkpoints, metrics scripts,
reports, tests, CI, and explicit baseline comparisons.

**What does the project not claim?**  
It does not claim real-world cloud removal, atmospheric correction, or geospatial accuracy.
