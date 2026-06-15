# Repository Health Check

## Automated Checks

- Tests: `pytest -q` passes.
- Lint: `ruff check .` passes.
- CI: `.github/workflows/ci.yml` runs tests and lint on Python 3.11.

## Artifact Hygiene

- Downloaded data is ignored: `data/raw/`
- Processed data is ignored except `.gitkeep`: `data/processed/*`
- Checkpoints are ignored except `.gitkeep`: `outputs/checkpoints/*`
- Sample images are ignored except `.gitkeep`: `outputs/samples/*`
- Docker build context excludes data, checkpoints, and sample outputs through `.dockerignore`.

## Deployment Smoke

- API import smoke passes with:
  `PYTHONPATH=src python -c "from uvicorn.importer import import_from_string; app = import_from_string('satellite_diffusion_restoration.api.app:app'); print(app.title)"`
- Streamlit import smoke passes with:
  `PYTHONPATH=src python -c "import streamlit_app; print(streamlit_app.__name__)"`

## Reproducibility Caveats

- Synthetic data generation and EuroSAT splits are seeded.
- GPU/CPU differences can change training curves slightly.
- DDPM sampling is stochastic for DDPM mode and deterministic for DDIM mode.
- EuroSAT scripts require local data or an explicit `--download` flag.
- The deployable API expects `outputs/checkpoints/unet_eurosat.pt` unless `MODEL_CHECKPOINT`
  is set.
