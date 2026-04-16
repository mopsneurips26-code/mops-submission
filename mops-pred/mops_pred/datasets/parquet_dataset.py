"""Dataset backed by the HF ``datasets`` library for Parquet data.

Supports three modes:

  - **Local Parquet** — ``data_dir`` points to a directory with ``train/``
    and ``test/`` sub-dirs of ``.parquet`` shards (produced by
    ``DatasetsWriter``).  Data is memory-mapped via Arrow, so only the
    rows actually accessed are loaded into RAM.

  - **HF Hub (cached)** — ``data_dir`` is a repo ID (e.g. ``user/dataset``).
    The dataset is downloaded once, cached locally, and memory-mapped.

  - **HF Hub (streaming)** — same repo ID with ``streaming=True``.
    Samples are decoded on-the-fly without any download.

All modes yield the same dict schema as ``ClutterDataset`` /
``WebDatasetDataset``, so models and the training loop are unchanged.

Column encodings (produced by ``DatasetsWriter``):
  - **image / semantic / instance / part / is_partnet**: ``datasets.Image``
  - **affordance / depth / normal**: compressed ``.npz`` bytes (binary)
  - **image_id / asset_id**: string; **class_idx**: int32 (optional)
"""

import io
from pathlib import Path
from typing import List, Set

import numpy as np
import torch
import torchvision.transforms.v2 as T_v2
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torchvision import tv_tensors
from torchvision.transforms.functional import to_tensor

# Mask keys stored as lossless PNG via datasets.Image feature.
_PNG_MASKS = ("semantic", "instance", "part", "is_partnet")
# Array keys stored as compressed .npz bytes.
_NPZ_ARRAYS = ("affordance", "depth", "normal")


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------


def _build_transforms(augment: bool):
    """Construct spatial, colour, and normalisation transforms."""
    if augment:
        spatial = T_v2.Compose(
            [
                T_v2.RandomResizedCrop(
                    size=(224, 224), scale=(0.8, 1.0), antialias=True
                ),
                T_v2.RandomHorizontalFlip(p=0.5),
                T_v2.RandomRotation(degrees=10),
            ]
        )
        color = T_v2.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )
    else:
        spatial = T_v2.Resize(size=(224, 224), antialias=True)
        color = None
    normalize = T_v2.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return spatial, color, normalize


def _to_pil(val) -> Image.Image:
    """Coerce an Image-column value to PIL.

    ``datasets.Image`` auto-decodes to PIL when accessed; this helper
    handles the fallback case of raw bytes or struct dicts.
    """
    if isinstance(val, Image.Image):
        return val
    if isinstance(val, dict):
        return Image.open(io.BytesIO(val["bytes"]))
    return Image.open(io.BytesIO(val))


def _decode_row(
    row: dict, labels: Set[str], spatial, color, normalize
) -> dict:
    """Convert one HF-datasets row into the training-loop dict format."""
    # --- RGB image ---
    image = tv_tensors.Image(to_tensor(_to_pil(row["image"]).convert("RGB")))

    masks: dict = {}

    # --- PNG masks (datasets.Image → PIL) ---
    for name in _PNG_MASKS:
        val = row.get(name)
        if val is None:
            continue
        if name != "is_partnet" and name not in labels:
            continue
        arr = np.array(_to_pil(val))
        t = torch.from_numpy(arr)
        masks[name] = tv_tensors.Mask(
            t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)
        )

    # --- NPZ arrays (affordance / depth / normal) ---
    for name in _NPZ_ARRAYS:
        if name not in labels:
            continue
        raw = row.get(name)
        if raw is None:
            continue
        arr = np.load(io.BytesIO(raw))["data"]
        if name == "affordance":
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
        t = torch.from_numpy(arr)
        masks[name] = tv_tensors.Mask(
            t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0),
        )

    # --- Transforms ---
    image, masks = spatial(image, masks)
    if color:
        image = color(image)

    result: dict = {"image": normalize(image)}
    result.update(masks)

    # --- Metadata ---
    result["image_id"] = row["image_id"]
    if row.get("class_idx") is not None:
        result["class_label"] = torch.tensor(row["class_idx"], dtype=torch.long)

    return result


# -----------------------------------------------------------------------
# Map-style dataset (local Parquet or cached HF Hub download)
# -----------------------------------------------------------------------


class ParquetDataset(Dataset):
    """Map-style dataset backed by HF ``datasets`` (memory-mapped Arrow).

    Works with both local Parquet directories and HF Hub repo IDs
    (data is downloaded once and cached).
    """

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        labels: List[str] | None = None,
        augment: bool = False,
    ):
        super().__init__()
        split = "train" if train else "test"

        if Path(data_dir).is_dir():
            self._ds = load_dataset("parquet", data_dir=data_dir, split=split)
        else:
            # Assume HF Hub repo ID (e.g. "user/dataset-name").
            self._ds = load_dataset(data_dir, split=split)

        self._labels = set(labels) if labels else {"semantic"}
        self._spatial, self._color, self._normalize = _build_transforms(augment)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        return _decode_row(
            self._ds[idx],
            self._labels,
            self._spatial,
            self._color,
            self._normalize,
        )


# -----------------------------------------------------------------------
# Streaming dataset (HF Hub, no full download)
# -----------------------------------------------------------------------


class StreamingParquetDataset(TorchIterableDataset):
    """Streaming iterable dataset for HF Hub repos.

    Samples are decoded on-the-fly and never fully materialised on disk.
    Use ``num_workers=0`` in the DataLoader to avoid each worker streaming
    the entire dataset independently.
    """

    def __init__(
        self,
        repo_id: str,
        train: bool = True,
        labels: List[str] | None = None,
        augment: bool = False,
    ):
        super().__init__()
        split = "train" if train else "test"
        self._ds = load_dataset(repo_id, split=split, streaming=True)
        if train:
            self._ds = self._ds.shuffle(seed=42, buffer_size=1000)

        self._labels = set(labels) if labels else {"semantic"}
        self._spatial, self._color, self._normalize = _build_transforms(augment)

    def __iter__(self):
        for row in self._ds:
            yield _decode_row(
                row,
                self._labels,
                self._spatial,
                self._color,
                self._normalize,
            )
