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

### 3. Optional EuroSAT Benchmark

The project also includes an optional EuroSAT RGB dataset wrapper. It uses real EuroSAT
images as clean targets, applies the same synthetic corruption pipeline, and trains the
same residual U-Net baseline. EuroSAT is not downloaded during tests and is only used when
you run the EuroSAT scripts.

The benchmark still uses synthetic corruptions. It is a real-image restoration benchmark,
not proof that the model handles real clouds or atmospheric artifacts.

Current EuroSAT smoke benchmark result with `1000` train samples, `200` validation samples,
and `5` epochs:

- corrupted input: MSE `0.027718`, PSNR `15.57 dB`, SSIM `0.4592`
- restored model: MSE `0.004265`, PSNR `23.70 dB`, SSIM `0.5651`
- improvement: MSE delta `+0.023453`, PSNR delta `+8.13 dB`

### 4. Conditional DDPM Restoration

The repository includes an experimental conditional DDPM restoration model. This is not
unconditional image generation. The clean image is the target `x0`, the corrupted image is
the conditioning input, the model predicts diffusion noise, and restoration is sampled
iteratively from noise while conditioning on the corrupted image.

The residual U-Net remains the supervised baseline. DDPM is not automatically better; it
only matters if it beats the corrupted-input baseline and ideally approaches or beats the
U-Net.

Current EuroSAT DDPM result after a small CPU run with `1000` train samples, `200` eval
samples, `10` epochs, EMA, DDIM sampling, and `100` diffusion timesteps:

- corrupted input: MSE `0.027718`, PSNR `15.57 dB`, SSIM `0.4592`
- DDPM one-step x0 t=`10`: MSE `0.001964`, PSNR `27.07 dB`, SSIM `0.5954`
- DDPM sampled restored: MSE `0.086399`, PSNR `10.63 dB`, SSIM `0.0229`
- U-Net restored: MSE `0.004265`, PSNR `23.70 dB`, SSIM `0.5651`

The one-step result is diagnostic only because it uses `x_t` generated from the clean image.
The deployable sampled DDPM is functional but not good yet. It currently underperforms both
the corrupted input and the residual U-Net.

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

## Train U-Net On EuroSAT

Download EuroSAT and train the residual U-Net:

```bash
python scripts/train_unet_eurosat.py --download
```

Useful smaller/faster run:

```bash
python scripts/train_unet_eurosat.py --download --max-train-samples 1000 --max-val-samples 200 --epochs 5
```

Expected outputs:

- best checkpoint: `outputs/checkpoints/unet_eurosat.pt`
- epoch sample grids: `outputs/samples/unet_eurosat_epoch_01.png` and later epoch files
- printed corrupted-input metrics, restored-model metrics, and MSE/PSNR deltas

If EuroSAT is already downloaded, omit `--download`.

## Evaluate EuroSAT U-Net

```bash
python scripts/evaluate_unet_eurosat.py
```

Useful smaller/faster evaluation:

```bash
python scripts/evaluate_unet_eurosat.py --max-samples 200
```

Expected outputs:

- final corrupted-input baseline metrics
- final restored-model metrics
- MSE and PSNR improvement deltas
- sample grids such as `outputs/samples/unet_eurosat_eval_00.png`

If the dataset is missing, run:

```bash
python scripts/train_unet_eurosat.py --download
```

## Train DDPM On Synthetic Data

Debug tiny DDPM overfit first:

```bash
python scripts/overfit_ddpm_tiny.py
```

Run one-step/full-sampling reconstruction diagnostics:

```bash
python scripts/diagnose_ddpm_reconstruction.py
```

Train the synthetic DDPM smoke model:

```bash
python scripts/train_ddpm_synthetic.py
```

Expected outputs:

- checkpoint: `outputs/checkpoints/ddpm_synthetic.pt`
- sample grids: `outputs/samples/ddpm_synthetic_epoch_01.png` and later epoch files
- printed corrupted-input metrics and DDPM sampled-restoration metrics

This is a correctness smoke test, not a quality benchmark.

## Train DDPM On EuroSAT

```bash
python scripts/train_ddpm_eurosat.py --download
```

If EuroSAT is already downloaded:

```bash
python scripts/train_ddpm_eurosat.py --max-train-samples 1000 --max-val-samples 200 --epochs 10
```

Expected outputs:

- checkpoint: `outputs/checkpoints/ddpm_eurosat.pt`
- sample grids: `outputs/samples/ddpm_eurosat_epoch_01.png` and later epoch files
- printed corrupted-input metrics and DDPM sampled-restoration metrics
- EMA weights saved in the checkpoint by default

Per-epoch sampled validation uses a small subset by default because reverse diffusion is
much slower than direct U-Net inference.

## Evaluate EuroSAT DDPM

```bash
python scripts/evaluate_ddpm_eurosat.py
```

Useful smaller/faster evaluation:

```bash
python scripts/evaluate_ddpm_eurosat.py --max-samples 200
```

If `outputs/checkpoints/unet_eurosat.pt` exists, the evaluator also prints U-Net metrics on
the same deterministic subset. It also prints one-step `x0` reconstruction diagnostics at
selected timesteps.

## Quality Checks

```bash
pytest -q
ruff check .
```

## Limitations

- synthetic runs are still smoke tests, not real benchmarks
- EuroSAT runs use real clean images but still use synthetic corruptions
- the default U-Net runs are intentionally small enough for local iteration
- the DDPM implementation is experimental and currently underperforms the U-Net baseline
- no FastAPI service yet
- no Streamlit demo yet
- no Kaggle credentials or external non-EuroSAT data are required
