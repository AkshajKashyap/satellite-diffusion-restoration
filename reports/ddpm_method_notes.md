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

## Meaningful Win Criteria

Before DDPM should be considered useful for this project, it should:

- beat the corrupted-input baseline on MSE and PSNR
- get close to or beat the residual U-Net baseline
- improve visual samples without washing out real EuroSAT structure
- maintain reproducible training and evaluation commands
- show gains on held-out real EuroSAT images with synthetic corruptions

The current DDPM implementation is functional but does not meet those quality criteria yet.
