# EuroSAT U-Net Benchmark Results

## Run Status

EuroSAT download, training, and evaluation succeeded in this environment.

## Dataset Size Used

- train samples: `1000`
- validation samples during training: `200`
- fresh deterministic evaluation samples: `200`
- image size: `64x64`
- data root: `data/raw`

## Training Config

- model: residual U-Net
- base channels: `16`
- residual scale: `0.5`
- epochs: `5`
- batch size: `16`
- learning rate: `1e-3`
- device used: `cuda`
- best checkpoint epoch: `4`
- checkpoint: `outputs/checkpoints/unet_eurosat.pt`

## Training Validation Metrics

Best checkpoint was saved at epoch `4`.

- corrupted input: MSE `0.029718`, PSNR `15.27 dB`, SSIM `0.4504`
- restored model: MSE `0.004128`, PSNR `23.84 dB`, SSIM `0.5771`
- improvement: MSE delta `+0.025590`, PSNR delta `+8.57 dB`

## Fresh Evaluation Metrics

Command:

```bash
python scripts/evaluate_unet_eurosat.py --max-samples 200
```

Results:

- corrupted input: MSE `0.027718`, PSNR `15.57 dB`, SSIM `0.4592`
- restored model: MSE `0.004265`, PSNR `23.70 dB`, SSIM `0.5651`
- improvement: MSE delta `+0.023453`, PSNR delta `+8.13 dB`

## Sample Output Paths

- `outputs/samples/unet_eurosat_epoch_01.png`
- `outputs/samples/unet_eurosat_epoch_02.png`
- `outputs/samples/unet_eurosat_epoch_03.png`
- `outputs/samples/unet_eurosat_epoch_04.png`
- `outputs/samples/unet_eurosat_epoch_05.png`
- `outputs/samples/unet_eurosat_eval_00.png`
- `outputs/samples/unet_eurosat_eval_01.png`
- `outputs/samples/unet_eurosat_eval_02.png`
- `outputs/samples/unet_eurosat_eval_03.png`

## Interpretation

The residual U-Net clearly beats the corrupted-input baseline on real EuroSAT RGB images
with synthetic corruptions. The improvement is large on MSE and PSNR, and SSIM improves as
well. This confirms that the supervised restoration baseline works beyond the synthetic
image generator.

This is still not a real cloud-removal benchmark. The clean images are real EuroSAT images,
but the corruptions are synthetic and may not match real cloud cover, haze, atmospheric
effects, or sensor-specific degradation.
