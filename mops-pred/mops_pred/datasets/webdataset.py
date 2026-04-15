"""WebDataset-based dataset for the training pipeline.

Reads sharded TAR archives produced by ``WebDatasetWriter`` and yields
samples in the same dict format as ``ClutterDataset``, so models and
the training loop remain unchanged.
"""

import io
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms.v2 as T_v2
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import tv_tensors
from torchvision.transforms.functional import to_tensor

# Mask keys stored as lossless PNG in the TAR shards.
_PNG_MASKS = ("semantic", "instance", "part", "is_partnet")
# Array keys stored as compressed .npz.
_NPZ_ARRAYS = ("affordance", "depth", "normal")


class WebDatasetDataset(IterableDataset):
    """Reads sharded TAR archives (WebDataset format) and yields samples
    matching the dict schema of ``ClutterDataset``.

    Each sample contains at minimum ``image`` (normalized tensor) and
    ``image_id`` (str).  Masks / arrays are included based on *labels*.

    Use ``dataset.name: webdataset`` in the config and point ``data_dir``
    to the root directory containing ``train/`` and ``test/`` sub-dirs
    with ``.tar`` shards.
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
        tar_dir = Path(data_dir) / split
        self._urls = sorted(str(p) for p in tar_dir.glob("*.tar"))
        if not self._urls:
            raise FileNotFoundError(f"No TAR shards found in {tar_dir}")

        self._train = train
        self._labels = set(labels) if labels else {"semantic"}

        # Read dataset length from dataset_info.json so Lightning can
        # compute estimated_stepping_batches for LR schedulers.
        info_path = Path(data_dir) / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            self._length = info["splits"][split]["num_images"]
        else:
            self._length = None

        # --- transforms (identical to ClutterDataset) ---
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

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode(self, sample: dict) -> dict:
        """Decode raw webdataset sample bytes and apply transforms."""
        # --- RGB image ---
        image = tv_tensors.Image(
            to_tensor(np.array(Image.open(io.BytesIO(sample["png"]))))
        )

        masks = {}

        # --- PNG masks (single-channel integer) ---
        for name in _PNG_MASKS:
            key = f"{name}.png"
            if key not in sample:
                continue
            # Always load is_partnet when present; others only if requested.
            if name != "is_partnet" and name not in self._labels:
                continue
            arr = np.array(Image.open(io.BytesIO(sample[key])))
            t = torch.from_numpy(arr)
            masks[name] = tv_tensors.Mask(
                t.unsqueeze(0) if t.ndim == 2 else t.permute(2, 0, 1)
            )

        # --- NPZ arrays (multi-channel float) ---
        for name in _NPZ_ARRAYS:
            key = f"{name}.npz"
            if key not in sample or name not in self._labels:
                continue
            arr = np.load(io.BytesIO(sample[key]))["data"]
            # Affordance masks are binary (0/1) — keep as uint8 to match
            # the H5 loader and avoid blowing up GPU memory with float64.
            # Continuous arrays (depth, normal) stay float32.
            if name == "affordance":
                arr = arr.astype(np.uint8)
            else:
                arr = arr.astype(np.float32)
            t = torch.from_numpy(arr)
            masks[name] = tv_tensors.Mask(
                t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0),
            )

        # --- Spatial + colour transforms ---
        image, masks = self._spatial(image, masks)
        if self._color:
            image = self._color(image)

        result: dict = {"image": self._normalize(image)}
        result.update(masks)

        # --- Metadata ---
        raw = sample["json"]
        meta = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
        result["image_id"] = meta["image_id"]
        if "class_idx" in meta:
            result["class_label"] = torch.tensor(meta["class_idx"], dtype=torch.long)

        return result

    def __len__(self):
        if self._length is None:
            raise TypeError(
                "Dataset length unknown — provide dataset_info.json in the data root."
            )
        return self._length

    def __iter__(self):
        # Build the pipeline fresh so each DataLoader worker gets its own
        # shard split (wds handles this via get_worker_info internally).
        pipeline = wds.WebDataset(self._urls, shardshuffle=1000 if self._train else 0)
        if self._train:
            pipeline = pipeline.shuffle(1000)
        pipeline = pipeline.map(self._decode)
        return iter(pipeline)
