# Conditional DDPM Method Notes

## Forward Diffusion

The clean target image is `x0`. During training, the scheduler samples a timestep `t` and
adds Gaussian noise with the closed-form DDPM forward process:

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
```

The model sees `x_t`, the timestep `t`, and the corrupted conditioning image.

## Noise Prediction Objective

The conditional U-Net predicts the noise `epsilon` that was used to create `x_t`. Training
uses MSE loss:

```text
loss = mse(predicted_epsilon, epsilon)
```

This is different from the residual U-Net baseline, which directly predicts a corrected
image or residual in one forward pass.

## Conditioning On The Corrupted Image

This project implements conditional restoration, not unconditional image generation. The
model input concatenates:

- noisy clean target at timestep `t`: `x_t`
- corrupted image: conditioning input

The concatenated tensor has 6 RGB channels. A timestep embedding is injected into each U-Net
block so the model can adapt its denoising behavior to the current noise level.

## U-Net Restoration vs DDPM Restoration

The residual U-Net is a direct supervised restoration model:

```text
restored = clamp(corrupted + predicted_residual, 0, 1)
```

It makes one prediction and is easy to train with MSE against the clean image.

The DDPM model is iterative. It starts from random noise and repeatedly applies the learned
reverse process while conditioning on the corrupted image. This is more flexible, but it is
slower and harder to train. A low noise-prediction loss does not automatically mean the
sampled restoration will beat the corrupted input.

## Why DDPM Is Harder And Slower

- Sampling requires one model forward pass per reverse timestep.
- The model must learn a denoising trajectory, not just a direct correction.
- Small timestep counts are faster but can produce weaker diffusion behavior.
- Small portfolio-scale training runs may underfit the reverse process badly.
- Evaluating sampled restorations is much more expensive than evaluating a direct U-Net.

## Practical Sampling And EMA

The training objective is still DDPM noise prediction. For practical evaluation, the repo
also supports deterministic DDIM-style sampling:

```bash
python scripts/evaluate_ddpm_eurosat.py --sampler ddim --sample-steps 25
```

DDIM is a sampling shortcut, not a new training objective. DDPM training scripts also keep
an exponential moving average of model weights by default and use the EMA model for
sampling/evaluation when available.

## Meaningful Win Criteria

Before DDPM should be considered useful for this project, it should:

- beat the corrupted-input baseline on MSE and PSNR
- get close to or beat the residual U-Net baseline
- improve visual samples without washing out real EuroSAT structure
- maintain reproducible training and evaluation commands
- show gains on held-out real EuroSAT images with synthetic corruptions

The current DDPM implementation is functional but does not meet those quality criteria yet.

## DDPM Debugging Findings

This milestone added three debugging tools:

- `scripts/overfit_ddpm_tiny.py`
- `scripts/diagnose_ddpm_reconstruction.py`
- one-step `x0` diagnostics in `scripts/evaluate_ddpm_eurosat.py`

Findings:

- Tiny overfit succeeded: fixed-batch noise loss dropped from `0.992377` to `0.097457`.
- Synthetic one-step reconstruction at low noise improved over corrupted input after the
  updated synthetic run: t=`10` MSE `0.011391` vs corrupted MSE `0.017860`.
- EuroSAT one-step reconstruction at low noise was strong: t=`10` MSE `0.001964`, PSNR
  `27.07 dB`, SSIM `0.5954`.
- EuroSAT high-noise one-step reconstruction still failed: t=`90` MSE `0.091859`, PSNR
  `10.37 dB`.
- EuroSAT full sampled DDPM still failed: MSE `0.086399`, PSNR `10.63 dB`, SSIM `0.0229`.
- DDPM does not beat the corrupted input and does not beat the residual U-Net.

Interpretation: the implementation can learn the noise-prediction objective, especially at
low noise levels, but the current training budget/model/sampling setup does not learn a
usable high-noise denoising trajectory from random noise.
