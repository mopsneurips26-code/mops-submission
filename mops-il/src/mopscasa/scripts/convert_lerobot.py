"""CLI: Convert a single HDF5 demonstration dataset to LeRobot format.

Usage::

    python src/mopscasa/scripts/convert_lerobot.py --datasets <path.hdf5> --output_path <out>
"""

import draccus

from mopscasa.lerobot_conversion.conv_config import ConversionConfig
from mopscasa.lerobot_conversion.conv_pipeline import create_lerobot_dataset

if __name__ == "__main__":
    cfg = draccus.parse(config_class=ConversionConfig)
    create_lerobot_dataset(cfg)
