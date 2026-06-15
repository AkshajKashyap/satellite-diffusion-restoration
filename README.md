# Satellite Diffusion Restoration

Satellite image restoration benchmark with synthetic corruptions, EuroSAT data, a strong
residual U-Net baseline, an experimental conditional DDPM, reports, FastAPI inference, and
a Streamlit demo.

**Status:** the residual U-Net is the deployable restoration baseline. The conditional DDPM
is implemented for research/debugging, but sampled DDPM restoration currently underperforms
both the corrupted input and the U-Net.

Suggested GitHub repo description:

```text
Satellite image restoration benchmark with synthetic corruptions, EuroSAT data, residual U-Net baseline, experimental conditional DDPM, metrics, reports, API, and Streamlit demo.
```

## Key Results

EuroSAT RGB images with synthetic corruptions, `200` sample evaluation:

| Method | MSE | PSNR | SSIM | Status |
| --- | ---: | ---: | ---: | --- |
| Corrupted input | 0.027718 | 15.57 dB | 0.4592 | Baseline |
| Residual U-Net | 0.004265 | 23.70 dB | 0.5651 | Deployable default |
| Sampled DDPM | 0.086399 | 10.63 dB | 0.0229 | Experimental, underperforms |

DDPM one-step x0 at t=`10` reached MSE `0.001964`, PSNR `27.07 dB`, and SSIM `0.5954`,
but this is diagnostic only because it uses `x_t` generated from the clean image. It is not
a deployable restoration result.

This project is about **satellite image restoration under synthetic corruptions**. It does
not claim real-world cloud removal.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
ruff check .
```

Run the synthetic data smoke test:

```bash
python scripts/smoke_data_pipeline.py
```

Makefile shortcuts are also available:

```bash
make install
make test
make lint
make smoke-data
make train-unet-synthetic
make eval-unet-eurosat
```

## U-Net Baseline

Train on EuroSAT:

```bash
python scripts/train_unet_eurosat.py --download --max-train-samples 1000 --max-val-samples 200 --epochs 5
```

Evaluate:

```bash
python scripts/evaluate_unet_eurosat.py --max-samples 200
```

Expected deployable checkpoint:

```text
outputs/checkpoints/unet_eurosat.pt
```

The residual U-Net predicts:

```text
restored = clamp(corrupted + predicted_residual, 0, 1)
```

Residual learning matters because the corrupted input is already close to the target, so
the model learns a correction instead of relearning the whole image.

## DDPM Experimental Path

The conditional DDPM is not unconditional generation. It predicts diffusion noise from:

- noised clean target `x_t`
- timestep embedding
- corrupted conditioning image

Debug and train:

```bash
python scripts/overfit_ddpm_tiny.py
python scripts/diagnose_ddpm_reconstruction.py
python scripts/train_ddpm_synthetic.py
python scripts/train_ddpm_eurosat.py --download --max-train-samples 1000 --max-val-samples 200 --epochs 10
python scripts/evaluate_ddpm_eurosat.py --max-samples 200
```

DDPM is useful in this repo as a research component and negative-result case study. It must
beat the corrupted-input baseline, and ideally the U-Net, before it should be treated as a
restoration model.

## API

The API serves the residual U-Net only.

```bash
PYTHONPATH=src uvicorn satellite_diffusion_restoration.api.app:app --host 127.0.0.1 --port 8000
```

Client smoke helper:

```bash
python scripts/smoke_api.py
```

Docker API image:

```bash
docker build -t satellite-restoration-api .
docker run --rm -p 8000:8000 satellite-restoration-api
```

The Docker image excludes downloaded EuroSAT data, checkpoints, and generated samples by
default. Mount or copy a trained U-Net checkpoint if you want `/restore` to run inside the
container.

Endpoints:

- `GET /health`
- `POST /restore` with an uploaded image file

If `outputs/checkpoints/unet_eurosat.pt` is missing, train it with:

```bash
python scripts/train_unet_eurosat.py --download
```

## Streamlit Demo

```bash
PYTHONPATH=src streamlit run streamlit_app.py
```

The demo lets you upload an image, run U-Net restoration, and view the known EuroSAT
benchmark table. DDPM is labeled experimental and is not the default demo model.

## Reports

- `reports/model_card.md`
- `reports/project_summary.md`
- `reports/interview_notes.md`
- `reports/repo_health_check.md`
- `reports/final_results.json`
- `reports/unet_eurosat_plan.md`
- `reports/unet_eurosat_results.md`
- `reports/ddpm_method_notes.md`
- `reports/ddpm_eurosat_results.md`

## Repo Structure

```text
src/satellite_diffusion_restoration/
  data/          synthetic data, EuroSAT wrapper, corruptions
  evaluation/    metrics and visualization helpers
  models/        residual U-Net, conditional U-Net, DDPM scheduler
  training/      U-Net and diffusion training/evaluation loops
  api/           FastAPI app
  inference.py   deployable U-Net inference helpers
scripts/         smoke, train, evaluate, and diagnostic commands
reports/         model card and experiment summaries
tests/           fast tests with no dataset downloads
```

## Limitations

- Benchmarks use synthetic corruptions, even on real EuroSAT images.
- This does not claim real-world cloud removal, atmospheric correction, or geospatial
  accuracy.
- EuroSAT images are small RGB patches, not full satellite products.
- The DDPM sampled restoration path is experimental and currently poor.
- Generated checkpoints, downloaded data, and sample images are local artifacts and should
  not be committed.
