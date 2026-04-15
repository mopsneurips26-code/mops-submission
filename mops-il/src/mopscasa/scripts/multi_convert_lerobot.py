import dataclasses
import glob
from pathlib import Path

import draccus
from loguru import logger

from mopscasa.lerobot_conversion.conv_config import ConversionConfig
from mopscasa.lerobot_conversion.conv_pipeline import create_lerobot_dataset


@dataclasses.dataclass
class MultiConversionConfig(ConversionConfig):
    # Root dir for globbing hdf5 datasets
    datasets: Path
    output_path: Path
    filter_terms: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        # glob all hdf5 files in the root dir
        self.datasets = sorted(
            glob.glob(str(self.datasets / "**/*.hdf5"), recursive=True)
        )
        if self.filter_terms:
            self.datasets = [
                ds
                for ds in self.datasets
                if not any(term in ds for term in self.filter_terms)
            ]
        logger.info(f"Found {len(self.datasets)} datasets for conversion.")
        super().__post_init__()


if __name__ == "__main__":
    cfg = draccus.parse(config_class=MultiConversionConfig)
    create_lerobot_dataset(cfg)
