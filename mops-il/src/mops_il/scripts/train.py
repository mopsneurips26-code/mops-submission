"""CLI entry point for MOPS-IL training.

Usage::

    python src/mops_il/scripts/train.py --dataset_path data/mopscasa_single_v2 --policy mopsflow

See ``MopsConfigCLI`` for all available flags.
"""

import draccus

from mops_il.config.cli_config import MopsConfigCLI
from mops_il.train import train

if __name__ == "__main__":
    cli_args = draccus.parse(config_class=MopsConfigCLI)
    config = cli_args.create_config()
    train(config)
