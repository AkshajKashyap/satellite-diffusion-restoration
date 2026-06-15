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
- epochs: `5`
- batch size: `16`
- learning rate: `3e-4`
- device used for this run: `cpu`
- checkpoint: `outputs/checkpoints/ddpm_eurosat.pt`

## Training-Epoch Sampled Metrics

Epoch 5 sampled validation subset:

- corrupted input: MSE `0.028888`, PSNR `15.39 dB`, SSIM `0.4689`
- DDPM restored: MSE `0.078653`, PSNR `11.04 dB`, SSIM `0.0279`
- DDPM improvement vs corrupted: MSE delta `-0.049765`, PSNR delta `-4.35 dB`

The noise-prediction loss dropped from `0.939075` to `0.251612`, but sampled restoration
quality remained poor.

## Fresh Evaluation Metrics

Command:

```bash
python scripts/evaluate_ddpm_eurosat.py --max-samples 200
```

Results:

- corrupted input: MSE `0.027718`, PSNR `15.57 dB`, SSIM `0.4592`
- DDPM restored: MSE `0.081368`, PSNR `10.90 dB`, SSIM `0.0264`
- DDPM improvement vs corrupted: MSE delta `-0.053650`, PSNR delta `-4.68 dB`
- U-Net restored: MSE `0.004265`, PSNR `23.70 dB`, SSIM `0.5651`
- DDPM vs U-Net: MSE delta `-0.077103`, PSNR delta `-12.81 dB`

## Sample Output Paths

- `outputs/samples/ddpm_eurosat_epoch_01.png`
- `outputs/samples/ddpm_eurosat_epoch_02.png`
- `outputs/samples/ddpm_eurosat_epoch_03.png`
- `outputs/samples/ddpm_eurosat_epoch_04.png`
- `outputs/samples/ddpm_eurosat_epoch_05.png`
- `outputs/samples/ddpm_eurosat_eval_00.png`
- `outputs/samples/ddpm_eurosat_eval_01.png`
- `outputs/samples/ddpm_eurosat_eval_02.png`
- `outputs/samples/ddpm_eurosat_eval_03.png`

## Honest Interpretation

The DDPM implementation is mechanically working: noise-prediction loss decreases, sampling
runs, checkpoints load, and evaluation compares corrupted input, DDPM output, and U-Net
output. However, the current DDPM is not a useful restoration model yet. It performs worse
than returning the corrupted input and far worse than the residual U-Net.

Most likely next fixes:

- train longer, preferably on GPU
- increase model capacity after the small implementation is stable
- tune the beta schedule for short timestep counts
- try more timesteps for training and sampling
- add EMA weights for sampling
- add a faster validation mode for frequent checks and reserve full sampling for final eval
- consider a residual or image-space auxiliary loss only as an experiment, while preserving
  the DDPM noise-prediction objective
