"""Parquet-based dataset for the training pipeline.

Reads sharded Parquet files produced by ``ParquetWriter`` and yields
samples in the same dict format as ``ClutterDataset`` / ``WebDatasetDataset``,
so models and the training loop remain unchanged.

Key encoding differences from WebDataset:
  - **affordance**: COCO RLE dicts (decoded via ``pycocotools``) instead of NPZ.
  - **depth / normal**: raw float32 bytes + shape/dtype struct instead of NPZ.
  - **images / masks**: PNG bytes (same as WebDataset).
"""

import io
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms.v2 as T_v2
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.functional import to_tensor

# Mask keys stored as lossless PNG.
_PNG_MASKS = ("semantic", "instance", "part", "is_partnet")


class ParquetDataset(Dataset):
    """Reads sharded Parquet files and yields samples matching the dict
    schema used by ``ClutterDataset`` and ``WebDatasetDataset``.

    Each sample contains at minimum ``image`` (normalised tensor) and
    ``image_id`` (str).  Masks / arrays are included based on *labels*.

    Point ``data_dir`` to the root directory containing ``train/`` and
    ``test/`` sub-dirs with ``.parquet`` shards.
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
        shard_dir = Path(data_dir) / split
        shard_paths = sorted(shard_dir.glob("*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"No Parquet shards found in {shard_dir}")

        self._table = pq.read_table(shard_dir)
        self._train = train
        self._labels = set(labels) if labels else {"semantic"}

        # --- transforms (identical to ClutterDataset / WebDatasetDataset) ---
        if augment:
            self._spatial = T_v2.Compose(
                [
                    T_v2.RandomResizedCrop(
                        size=(224, 224), scale=(0.8, 1.0), antialias=True
                    ),
                    T_v2.RandomHorizontalFlip(p=0.5),
                    T_v2.RandomRotation(degrees=10),
                ]
            )
            self._color = T_v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
        else:
            self._spatial = T_v2.Resize(size=(224, 224), antialias=True)
            self._color = None

        self._normalize = T_v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self._table)

    def __getitem__(self, idx: int) -> dict:
        row = {col: self._table.column(col)[idx].as_py() for col in self._table.column_names}

        # --- RGB image ---
        image = tv_tensors.Image(
            to_tensor(np.array(Image.open(io.BytesIO(row["image"]))))
        )

        masks = {}

        # --- PNG masks (single-channel integer) ---
        for name in _PNG_MASKS:
            raw = row.get(name)
            if raw is None:
                continue
            if name != "is_partnet" and name not in self._labels:
                continue
            arr = np.array(Image.open(io.BytesIO(raw)))
            t = torch.from_numpy(arr)
            masks[name] = tv_tensors.Mask(
                t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)
            )

        # --- Affordance (COCO RLE) ---
        if "affordance" in self._labels and row.get("affordance") is not None:
            rles = row["affordance"]
            channels = np.stack(
                [coco_mask.decode({"size": r["size"], "counts": r["counts"].encode("ascii")}) for r in rles],
                axis=-1,
            ).astype(np.uint8)
            t = torch.from_numpy(channels).permute(2, 0, 1)
            masks["affordance"] = tv_tensors.Mask(t)

        # --- Depth / Normal (raw float bytes) ---
        for name in ("depth", "normal"):
            if name not in self._labels:
                continue
            cell = row.get(name)
            if cell is None:
                continue
            arr = np.frombuffer(cell["data"], dtype=cell["dtype"]).reshape(cell["shape"])
            arr = arr.astype(np.float32)
            t = torch.from_numpy(arr.copy())
            masks[name] = tv_tensors.Mask(
                t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)
            )

        # --- Spatial + colour transforms ---
        image, masks = self._spatial(image, masks)
        if self._color:
            image = self._color(image)

        result: dict = {"image": self._normalize(image)}
        result.update(masks)

        # --- Metadata ---
        result["image_id"] = row["image_id"]
        if row.get("class_idx") is not None:
            result["class_label"] = torch.tensor(row["class_idx"], dtype=torch.long)

        return result
