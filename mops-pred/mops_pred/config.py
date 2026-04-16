"""Experiment configuration dataclasses (draccus-based)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WandbConfig:
    project: str = "mops-pred-2026"


@dataclass
class DatasetConfig:
    name: str = "object_centric"
    data_dir: str = "data/mops_data/mops_object_dataset"
    test_dir: Optional[str] = None
    num_classes: int = 46
    labels: Optional[list[str]] = None
    alias: Optional[str] = None
    streaming: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 40


@dataclass
class ModelConfig:
    name: str = "segmentation"
    num_classes: int = 46
    backbone: Optional[str] = None
    task: Optional[str] = None
    multilabel: Optional[bool] = None
    loss: Optional[str] = None
    partnet_iou: Optional[bool] = None


@dataclass
class ExperimentConfig:
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    mode: str = "train"  # train, debug, test
    checkpoint: Optional[str] = None
