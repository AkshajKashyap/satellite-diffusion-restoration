# Model Card

## Project Goal

This project studies satellite image restoration under synthetic corruptions. It builds a
clean benchmark around generated synthetic images and EuroSAT RGB image patches, then
compares a deployable residual U-Net baseline with an experimental conditional DDPM.

## Data Used

- Synthetic pseudo-satellite RGB tensors for tests and smoke runs
- EuroSAT RGB image patches loaded through `torchvision.datasets.EuroSAT`

EuroSAT images are used as clean targets. The degraded inputs are created by the project
corruption pipeline, not by real observed cloud labels.

## Corruption Setup

The corruption pipeline can apply:

- Gaussian noise
- random rectangular masks
- soft cloud-like blobs
- optional Gaussian blur

These corruptions are useful for controlled restoration experiments, but they do not prove
real-world cloud removal.

## Model Choices

### Residual U-Net

The residual U-Net predicts a correction to the corrupted image:

```text
restored = clamp(corrupted + predicted_residual, 0, 1)
```

This is the deployable baseline and the default inference/API/demo model.

### Conditional DDPM

The DDPM predicts diffusion noise conditioned on the corrupted image. It is included as an
experimental research component. It is not the default product path because full sampling
currently underperforms the simpler baseline.

## EuroSAT Results

| Method | MSE | PSNR | SSIM |
| --- | ---: | ---: | ---: |
| Corrupted input | 0.027718 | 15.57 dB | 0.4592 |
| Residual U-Net | 0.004265 | 23.70 dB | 0.5651 |
| Sampled DDPM experimental | 0.086399 | 10.63 dB | 0.0229 |

DDPM one-step x0 at t=10 reached MSE `0.001964`, PSNR `27.07 dB`, and SSIM `0.5954`, but
that is diagnostic only because it uses `x_t` generated from the clean image.

## Limitations

- This is satellite image restoration under synthetic corruptions.
- It does not claim real-world cloud removal.
- EuroSAT images are small RGB patches, not full satellite products.
- The DDPM sampled restoration path is currently not useful.
- Benchmarks are small and portfolio-scale.

## Ethical And Validity Caveats

Restored imagery can hallucinate or remove meaningful detail. These models should not be
used for safety-critical, legal, military, environmental, or disaster-response decisions.
The benchmark is useful for engineering practice and model comparison, not operational
remote-sensing validation.

## What This Project Does Not Claim

- It does not claim real cloud removal.
- It does not claim atmospheric correction.
- It does not claim geospatial accuracy.
- It does not claim state-of-the-art diffusion restoration.
- It does not replace domain-specific remote-sensing QA.
