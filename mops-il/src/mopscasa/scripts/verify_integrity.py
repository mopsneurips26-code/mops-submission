import dataclasses
import pathlib

import draccus
from loguru import logger

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass
class CliConfig:
    """Configuration for the CLI tool."""

    lr_path: pathlib.Path


if __name__ == "__main__":
    conf = draccus.parse(CliConfig)
    logger.info(f"Verifying integrity for dataset at: {conf.lr_path}")

    ds = LeRobotDataset(
        repo_id=str(conf.lr_path),
        root=conf.lr_path,
    )

    logger.info(f"Dataset contains {len(ds)} samples.")
    logger.info(f"Number of episodes: {ds.num_episodes}")
    logger.info(f"Number of tasks: {ds.meta.total_tasks}")
