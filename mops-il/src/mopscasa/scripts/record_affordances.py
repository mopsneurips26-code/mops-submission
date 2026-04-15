"""Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robocasa's dataset_states_to_obs.py script.
"""

import draccus

from mopscasa.image_recording import (
    RecorderConfig,
    dataset_states_to_obs_multiprocessing,
)

if __name__ == "__main__":
    config = draccus.parse(RecorderConfig)
    dataset_states_to_obs_multiprocessing(config)

    with open(config.output_name.with_suffix(".yaml"), "w") as f:
        draccus.dump(config, f)
