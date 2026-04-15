import dataclasses
import glob

import draccus
from loguru import logger

from mopscasa.image_recording import (
    RecorderConfig,
    dataset_states_to_obs_multiprocessing,
)


@dataclasses.dataclass
class AllDatasetToAffordsConfig:
    root_dir: str
    suffix: str = "mops"
    num_procs: int = 5
    filter_key: str = None


if __name__ == "__main__":
    config = draccus.parse(config_class=AllDatasetToAffordsConfig)

    # find all hdf5 files in root_dir
    dataset_paths = glob.glob(f"{config.root_dir}/**/*.hdf5", recursive=True)
    logger.info(f"Found {len(dataset_paths)} datasets in {config.root_dir}")
    for i, path in enumerate(dataset_paths):
        if "/mg/" in path:
            continue

        logger.info(f"Processing dataset {i + 1}/{len(dataset_paths)}: {path}")

        single_conf = RecorderConfig(
            dataset=path,
            suffix=config.suffix,
            filter_key=config.filter_key,
            num_procs=config.num_procs,
            copy_dones=False,
            copy_rewards=False,
        )
        dataset_states_to_obs_multiprocessing(single_conf)

        with open(single_conf.output_name.with_suffix(".yaml"), "w") as f:
            draccus.dump(single_conf, f)
        logger.info(f"Finished processing dataset {i + 1}/{len(dataset_paths)}: {path}")
