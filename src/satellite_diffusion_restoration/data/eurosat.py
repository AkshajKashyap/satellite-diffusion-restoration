"""EuroSAT RGB restoration dataset wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from satellite_diffusion_restoration.data.corruptions import CombinedCorruption, CorruptionConfig


class EuroSATRestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return ``(corrupted, clean)`` pairs from EuroSAT RGB images."""

    def __init__(
        self,
        root: str | Path = "data/raw",
        image_size: int = 64,
        download: bool = False,
        seed: int = 0,
        max_samples: int | None = None,
        corruption_config: CorruptionConfig | dict[str, Any] | None = None,
        indices: Sequence[int] | None = None,
        base_dataset: Dataset | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.seed = seed
        self.base_dataset = base_dataset or _load_torchvision_eurosat(self.root, download)

        selected_indices = list(range(len(self.base_dataset))) if indices is None else list(indices)
        if max_samples is not None:
            selected_indices = selected_indices[:max_samples]
        if not selected_indices:
            raise ValueError("EuroSATRestorationDataset received no samples")
        self.indices = selected_indices

        if isinstance(corruption_config, dict):
            config = CorruptionConfig(**corruption_config)
        else:
            config = corruption_config or CorruptionConfig()
        self.corruption_config = config
        self.corrupt = CombinedCorruption(config)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        base_index = self.indices[index]
        sample = self.base_dataset[base_index]
        image = sample[0] if isinstance(sample, tuple) else sample
        clean = image_to_rgb_tensor(image, self.image_size)

        generator = torch.Generator().manual_seed(self.seed + base_index)
        corrupted = self.corrupt(clean, generator=generator)
        return corrupted, clean


def deterministic_split_indices(
    num_items: int,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return deterministic train/validation indices for datasets without official splits."""
    if num_items < 2:
        raise ValueError("Need at least two samples to create a train/validation split")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(num_items, generator=generator).tolist()
    val_count = max(1, int(round(num_items * val_fraction)))
    val_count = min(val_count, num_items - 1)

    val_indices = shuffled[:val_count]
    train_indices = shuffled[val_count:]

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    if not train_indices:
        raise ValueError("Train split is empty; increase dataset size or max_train_samples")
    if not val_indices:
        raise ValueError("Validation split is empty; increase dataset size or max_val_samples")

    return train_indices, val_indices


def build_eurosat_train_val_datasets(
    root: str | Path = "data/raw",
    image_size: int = 64,
    download: bool = False,
    seed: int = 42,
    max_train_samples: int | None = 1000,
    max_val_samples: int | None = 200,
    val_fraction: float = 0.2,
    corruption_config: CorruptionConfig | dict[str, Any] | None = None,
) -> tuple[EuroSATRestorationDataset, EuroSATRestorationDataset]:
    """Build deterministic EuroSAT train/validation restoration datasets."""
    base_dataset = _load_torchvision_eurosat(Path(root), download)
    train_indices, val_indices = deterministic_split_indices(
        len(base_dataset),
        val_fraction=val_fraction,
        seed=seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    train_dataset = EuroSATRestorationDataset(
        root=root,
        image_size=image_size,
        download=False,
        seed=seed,
        corruption_config=corruption_config,
        indices=train_indices,
        base_dataset=base_dataset,
    )
    val_dataset = EuroSATRestorationDataset(
        root=root,
        image_size=image_size,
        download=False,
        seed=seed + 50_000,
        corruption_config=corruption_config,
        indices=val_indices,
        base_dataset=base_dataset,
    )
    return train_dataset, val_dataset


def image_to_rgb_tensor(image: object, image_size: int = 64) -> torch.Tensor:
    """Convert a PIL, NumPy, or tensor image to ``(3, H, W)`` float tensor in ``[0, 1]``."""
    if isinstance(image, torch.Tensor):
        tensor = _tensor_to_chw_float(image)
    else:
        tensor = _pil_or_array_to_tensor(image)

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")

    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.clamp(0.0, 1.0).contiguous()


def _load_torchvision_eurosat(root: Path, download: bool) -> Dataset:
    try:
        from torchvision.datasets import EuroSAT
    except Exception as exc:  # pragma: no cover - depends on local torchvision install
        raise ImportError(
            "torchvision is required for EuroSAT support. Install project requirements first."
        ) from exc

    try:
        return EuroSAT(root=str(root), download=download)
    except RuntimeError as exc:
        if not download:
            raise FileNotFoundError(
                f"EuroSAT was not found under {root}. Run "
                "`python scripts/train_unet_eurosat.py --download` to download it."
            ) from exc
        raise


def _pil_or_array_to_tensor(image: object) -> torch.Tensor:
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {array.shape}")

    tensor = torch.from_numpy(np.array(array, copy=True))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
    return tensor.permute(2, 0, 1)


def _tensor_to_chw_float(image: torch.Tensor) -> torch.Tensor:
    tensor = image.detach().clone()
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor image with 3 dimensions, got {tuple(tensor.shape)}")

    if tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1)

    tensor = tensor.float()
    if float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor
