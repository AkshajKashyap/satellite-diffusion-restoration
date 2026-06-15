"""Client-side smoke helper for a running restoration API server."""

from __future__ import annotations

from io import BytesIO

from PIL import Image
import requests


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    health_response = requests.get(f"{base_url}/health", timeout=10)
    print(f"GET /health -> {health_response.status_code}")
    print(health_response.json())

    image = Image.new("RGB", (64, 64), color=(80, 120, 70))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    files = {"file": ("synthetic.png", buffer.getvalue(), "image/png")}
    restore_response = requests.post(f"{base_url}/restore", files=files, timeout=30)
    print(f"POST /restore -> {restore_response.status_code}")
    print(f"Content-Type: {restore_response.headers.get('content-type')}")
    print(f"Response bytes: {len(restore_response.content)}")


if __name__ == "__main__":
    main()
