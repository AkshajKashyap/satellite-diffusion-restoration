# EuroSAT Conditional DDPM Results

## Run Status

The DDPM code path was implemented and executed on the local EuroSAT data.

## Dataset Size Used

- train samples: `1000`
- validation samples during training: `200`
- sampled validation subset per training epoch: `32`
- fresh deterministic evaluation samples: `200`
- image size: `64x64`
- data root: `data/raw`

## Training Config

- model: conditional U-Net noise predictor
- conditioning: concatenate `x_t` and corrupted image, 6 input channels
- prediction target: diffusion noise `epsilon`
- scheduler: linear beta DDPM
- timesteps: `100`
- epochs: `10`
- batch size: `16`
- learning rate: `3e-4`
- EMA: enabled, decay `0.995`
- sampler: deterministic DDIM
- sample steps: `25`
- device used for this run: `cpu`
- checkpoint: `outputs/checkpoints/ddpm_eurosat.pt`

## Training-Epoch Sampled Metrics

Epoch 10 sampled validation subset:

- corrupted input: MSE `0.028888`, PSNR `15.39 dB`, SSIM `0.4689`
- DDPM restored: MSE `0.085033`, PSNR `10.70 dB`, SSIM `0.0251`
- DDPM improvement vs corrupted: MSE delta `-0.056145`, PSNR delta `-4.69 dB`

The noise-prediction loss dropped from `0.939075` to `0.050278`, but sampled restoration
quality remained poor.

## Fresh Evaluation Metrics

Command:

```bash
python scripts/evaluate_ddpm_eurosat.py --max-samples 200
```

Results:

- corrupted input: MSE `0.027718`, PSNR `15.57 dB`, SSIM `0.4592`
- DDPM one-step x0 t=`10`: MSE `0.001964`, PSNR `27.07 dB`, SSIM `0.5954`
- DDPM one-step x0 t=`50`: MSE `0.029704`, PSNR `15.27 dB`, SSIM `0.1161`
- DDPM one-step x0 t=`90`: MSE `0.091859`, PSNR `10.37 dB`, SSIM `0.0420`
- DDPM sampled restored: MSE `0.086399`, PSNR `10.63 dB`, SSIM `0.0229`
- DDPM improvement vs corrupted: MSE delta `-0.058681`, PSNR delta `-4.94 dB`
- U-Net restored: MSE `0.004265`, PSNR `23.70 dB`, SSIM `0.5651`
- DDPM sampled vs U-Net: MSE delta `-0.082134`, PSNR delta `-13.07 dB`
- one-step t=`10` vs U-Net: MSE delta `+0.002301`, PSNR delta `+3.37 dB`

Important caveat: one-step `x0` diagnostics use `x_t` generated from the clean image, so
they are not deployable restoration results. They are a diagnostic for whether the model
learned noise prediction at a given timestep.

## Sample Output Paths

- `outputs/samples/ddpm_eurosat_epoch_01.png`
- `outputs/samples/ddpm_eurosat_epoch_02.png`
- `outputs/samples/ddpm_eurosat_epoch_03.png`
- `outputs/samples/ddpm_eurosat_epoch_04.png`
- `outputs/samples/ddpm_eurosat_epoch_05.png`
- `outputs/samples/ddpm_eurosat_epoch_06.png`
- `outputs/samples/ddpm_eurosat_epoch_07.png`
- `outputs/samples/ddpm_eurosat_epoch_08.png`
- `outputs/samples/ddpm_eurosat_epoch_09.png`
- `outputs/samples/ddpm_eurosat_epoch_10.png`
- `outputs/samples/ddpm_eurosat_eval_00.png`
- `outputs/samples/ddpm_eurosat_eval_01.png`
- `outputs/samples/ddpm_eurosat_eval_02.png`
- `outputs/samples/ddpm_eurosat_eval_03.png`
- `outputs/samples/ddpm_tiny_overfit_before.png`
- `outputs/samples/ddpm_tiny_overfit_after.png`
- `outputs/samples/ddpm_diagnostic_one_step_t10.png`
- `outputs/samples/ddpm_diagnostic_sampled.png`

## DDPM Debugging Findings

- Tiny overfit succeeded: noise loss dropped from `0.992377` to `0.097457`.
- Low-noise one-step reconstruction works well. On EuroSAT, t=`10` beat both corrupted
  input and U-Net on MSE/PSNR.
- Mid/high-noise one-step reconstruction does not work well enough.
- Full sampling from random noise is still much worse than the corrupted input.
- DDPM does not beat U-Net.

## Honest Interpretation

The DDPM implementation is mechanically working: noise-prediction loss decreases, sampling
runs, checkpoints load, EMA weights are saved/used, DDIM sampling works, and evaluation
compares corrupted input, one-step diagnostics, sampled DDPM output, and U-Net output.
However, the current DDPM is not a useful restoration model yet. It performs worse than
returning the corrupted input and far worse than the residual U-Net.

Most likely next fixes:

- train much longer, preferably on GPU
- increase model capacity after the small implementation is stable
- tune the beta schedule for short timestep counts
- try more timesteps for training and sampling
- tune or reduce EMA decay for short training runs
- investigate starting sampling from the corrupted image plus noise rather than pure noise
- consider a residual or image-space auxiliary loss only as an experiment, while preserving
  the DDPM noise-prediction objective
