.PHONY: install test lint smoke-data train-unet-synthetic eval-unet-eurosat api streamlit

PYTHON ?= python
PIP ?= pip
PYTHONPATH := src

install:
	$(PIP) install -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH) pytest -q

lint:
	ruff check .

smoke-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/smoke_data_pipeline.py

train-unet-synthetic:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_unet_baseline.py

eval-unet-eurosat:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/evaluate_unet_eurosat.py --max-samples 200

api:
	PYTHONPATH=$(PYTHONPATH) uvicorn satellite_diffusion_restoration.api.app:app --host 127.0.0.1 --port 8000

streamlit:
	PYTHONPATH=$(PYTHONPATH) streamlit run streamlit_app.py
