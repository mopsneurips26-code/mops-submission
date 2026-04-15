from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, IterableDataset

from mops_pred.config import DatasetConfig

_DATA_REPOSITORY = {}


def register_dataset(cls=None, *, name=None):
    def _register(cls):
        local_name = name
        if local_name is None:
            local_name = cls.__name__
        if local_name in _DATA_REPOSITORY:
            return cls
        _DATA_REPOSITORY[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def _is_h5(path: str) -> bool:
    return path.endswith((".h5", ".hdf5"))


def _is_parquet(data_dir: str) -> bool:
    """Return True if *data_dir* contains Parquet shards (train/*.parquet)."""
    train_dir = Path(data_dir) / "train"
    return train_dir.is_dir() and any(train_dir.glob("*.parquet"))


def create_dataloader(
    dataset_cfg: DatasetConfig,
    batch_size: int = 64,
    shuffle_train: bool = True,
    augment: bool = True,
):
    """Create train and test DataLoaders from a DatasetConfig.

    Format is auto-detected from ``data_dir``:
      * **.h5 / .hdf5 file** → legacy HDF5 reader selected by ``dataset_cfg.name``
      * **directory with .parquet shards** → Parquet reader (HF-native)
      * **directory with .tar shards** → WebDataset reader

    Args:
        dataset_cfg: Dataset configuration specifying name, paths, and labels.
        batch_size: Batch size for both loaders.
        shuffle_train: Whether to shuffle the training DataLoader.
        augment: Whether to apply data augmentation to the training split.

    Returns:
        Tuple of ``(train_loader, test_loader)``.
    """
    data_dir = dataset_cfg.data_dir
    test_dir = dataset_cfg.test_dir or data_dir

    if _is_h5(data_dir):
        # Legacy HDF5 path — use registered dataset class.
        cls = _DATA_REPOSITORY[dataset_cfg.name]
        train_ds = cls(data_dir, train=True, augment=augment, labels=dataset_cfg.labels)
        test_ds = cls(test_dir, train=False, labels=dataset_cfg.labels)
    elif _is_parquet(data_dir):
        # Parquet shards (HF-native format).
        from mops_pred.datasets.parquet_dataset import ParquetDataset

        train_ds = ParquetDataset(
            data_dir, train=True, augment=augment, labels=dataset_cfg.labels
        )
        test_ds = ParquetDataset(
            test_dir, train=False, labels=dataset_cfg.labels
        )
    else:
        # Default: directory → WebDataset shards.
        from mops_pred.datasets.webdataset import WebDatasetDataset

        train_ds = WebDatasetDataset(
            data_dir, train=True, augment=augment, labels=dataset_cfg.labels
        )
        test_ds = WebDatasetDataset(
            test_dir, train=False, labels=dataset_cfg.labels
        )

    # IterableDataset (WebDataset) handles shuffling internally.
    is_iterable = isinstance(train_ds, IterableDataset)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train and not is_iterable,
        num_workers=8,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return train_loader, test_loader
