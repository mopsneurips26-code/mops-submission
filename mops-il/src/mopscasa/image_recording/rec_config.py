import dataclasses
import pathlib

from loguru import logger


@dataclasses.dataclass
class RecorderConfig:
    # Path to input hdf5 dataset
    dataset: pathlib.Path
    # Automatic suffix for output dataset name
    suffix: str = "mops"
    # Name of output hdf5 dataset. Generates an output name if not provided.
    output_name: pathlib.Path | None = None
    # Filter key for input dataset
    filter_key: str = None
    # Stop after n trajectories are processed
    n: int | None = None
    # Use shaped rewards
    shaped: bool = False
    # Camera name(s) to use for image observations.
    # Leave out to not use image observations.
    camera_names: list[str] = dataclasses.field(
        default_factory=lambda: [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
    )
    # Height of image observations
    camera_height: int = 256
    # Width of image observations
    camera_width: int = 256
    # How to write done signal. If 0, done is 1 whenever s' is a success state.
    # If 1, done is 1 at the end of each trajectory. If 2, both.
    done_mode: int = 0
    # Copy rewards from source file instead of inferring them
    copy_rewards: bool = True
    # Copy dones from source file instead of re-writing them
    copy_dones: bool = True
    # Include next obs in dataset
    include_next_obs: bool = False
    # Disable compressing observations with gzip option in hdf5
    no_compress: bool = False
    # Number of parallel processes for extracting image obs
    num_procs: int = 5
    # Add datagen info (used for mimicgen)
    add_datagen_info: bool = False
    # Use generative textures
    generative_textures: bool = True
    # Randomize cameras
    randomize_cameras: bool = True
    # Stop a worker immediately on the first error (no retry)
    stop_on_error: bool = True
    # Max retries per demo index before giving up (None = unlimited)
    max_retries: int | None = None

    def __post_init__(self):
        self.dataset = pathlib.Path(self.dataset)
        if self.output_name is None:
            self.output_name = self._generate_output_name()

    def _generate_output_name(self) -> pathlib.Path:
        input_name = pathlib.Path(self.dataset)
        parts = list(input_name.parts)
        new_parts = []

        for _i, part in enumerate(parts):
            # replace subfolder v0.1 with v0.1_mops. can be somewhere in the path
            if part.startswith("v") and "." in part:
                new_parts.append(part + "_" + self.suffix)
                continue
            elif part == "mg" or part.startswith("2024"):
                continue
            else:
                new_parts.append(part)
        output_dir = pathlib.Path(*new_parts[:-1])
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self.filter_key if self.filter_key is not None else "demos"
        filename += "_mops"
        if self.generative_textures:
            filename += "_gentex"
        filename += f"_im{self.camera_width}"
        if self.randomize_cameras:
            filename += "_randcams"
        filename += ".hdf5"

        file_path = output_dir / filename
        logger.info(f"Generated output file path: {file_path}")
        return file_path
