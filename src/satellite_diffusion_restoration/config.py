"""Small configuration helpers for scripts and experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config file and overlay it on optional defaults."""
    config: dict[str, Any] = dict(defaults or {})
    if path is None:
        return config

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {config_path}")

    config.update(loaded)
    return config
