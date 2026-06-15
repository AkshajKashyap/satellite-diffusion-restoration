"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch, then return a PyTorch generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    return torch.Generator().manual_seed(seed)
