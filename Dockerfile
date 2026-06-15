FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/satellite_diffusion_restoration_matplotlib \
    MODEL_CHECKPOINT=outputs/checkpoints/unet_eurosat.pt \
    MODEL_DEVICE=auto

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "satellite_diffusion_restoration.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
