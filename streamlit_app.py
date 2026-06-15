"""Streamlit demo for the residual U-Net restoration baseline."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/satellite_diffusion_restoration_matplotlib")

from PIL import Image
import streamlit as st
import torch

from satellite_diffusion_restoration.inference import (
    DEFAULT_UNET_CHECKPOINT,
    MissingCheckpointError,
    restore_image,
    TRAIN_UNET_COMMAND,
)


BENCHMARK_ROWS = [
    {"Model": "Corrupted input", "MSE": 0.027718, "PSNR": 15.57, "SSIM": 0.4592},
    {"Model": "Residual U-Net", "MSE": 0.004265, "PSNR": 23.70, "SSIM": 0.5651},
    {"Model": "Sampled DDPM (experimental)", "MSE": 0.086399, "PSNR": 10.63, "SSIM": 0.0229},
]

name = "streamlit_app"


def main() -> None:
    st.set_page_config(page_title="Satellite Restoration", layout="wide")
    st.title("Satellite Image Restoration")
    st.write(
        "Residual U-Net restoration for satellite image restoration under synthetic corruptions. "
        "The DDPM path is included as an experimental research component, not the product default."
    )

    st.subheader("Known EuroSAT Benchmark")
    st.dataframe(BENCHMARK_ROWS, hide_index=True, use_container_width=True)
    st.caption(
        "EuroSAT uses real RGB image patches with synthetic corruptions. This project does not claim "
        "real-world cloud removal."
    )

    checkpoint_path = Path(os.environ.get("MODEL_CHECKPOINT", DEFAULT_UNET_CHECKPOINT))
    if not checkpoint_path.exists():
        st.warning(f"Missing U-Net checkpoint: `{checkpoint_path}`")
        st.code(TRAIN_UNET_COMMAND, language="bash")
        return

    uploaded = st.file_uploader("Upload an RGB satellite-style image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    image = Image.open(BytesIO(uploaded.read())).convert("RGB")
    left, right = st.columns(2)
    left.image(image, caption="Uploaded / corrupted input", use_container_width=True)

    if st.button("Run U-Net Restoration", type="primary"):
        try:
            device_name = os.environ.get("MODEL_DEVICE", "auto")
            if device_name == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif device_name == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
            else:
                device = torch.device(device_name)
            restored, _ = restore_image(image, checkpoint_path=checkpoint_path, device=device)
            right.image(restored, caption="Residual U-Net restoration", use_container_width=True)
        except MissingCheckpointError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
