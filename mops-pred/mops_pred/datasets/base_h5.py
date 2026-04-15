from abc import ABC, abstractmethod
from typing import Any, Dict

import h5py
import numpy as np
from torch.utils.data import Dataset


class BaseH5Dataset(Dataset, ABC):
    """
    Abstract base class for HDF5 datasets.

    Handles common functionalities like lazy file opening and index filtering
    based on training/testing splits.
    """

    def __init__(self, h5_path: str, train: bool, augment: bool):
        self.h5_path = h5_path
        self.is_train = train
        self.augment = augment
        self.h5_file = None  # Lazy loading: open in __getitem__
        self.indices = self._get_split_indices()

    def _get_split_indices(self) -> np.ndarray:
        """Reads split information from HDF5 and returns indices for the split."""
        with h5py.File(self.h5_path, "r") as f:
            # Determine the correct key for splits based on dataset structure
            if "metadata" in f and "splits" in f["metadata"]:
                splits = f["metadata"]["splits"][:]  # For clutter dataset
            elif "labels" in f and "splits" in f["labels"]:
                splits = f["labels"]["splits"][:]  # For object-centric dataset
            else:
                raise KeyError("Could not find 'splits' information in HDF5 file.")
            return np.where(splits == self.is_train)[0]

    def __len__(self) -> int:
        return len(self.indices)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Subclasses must implement this method."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        actual_idx = self.indices[idx]
        image_id = f"image_{actual_idx:06d}"

        return {"actual_idx": actual_idx, "image_id": image_id}
