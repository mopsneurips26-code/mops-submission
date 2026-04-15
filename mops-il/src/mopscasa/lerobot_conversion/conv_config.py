import dataclasses
import pathlib


@dataclasses.dataclass
class ConversionConfig:
    # Path to input hdf5 dataset
    datasets: pathlib.Path | list[pathlib.Path]
    # Name of output hdf5 dataset. Generates an output name if not provided.
    output_path: pathlib.Path | None = None
    # Stop after n trajectories are processed
    n: int | None = None
    # Height of image observations
    camera_height: int = 256
    # Width of image observations
    camera_width: int = 256

    def __post_init__(self):
        if isinstance(self.datasets, pathlib.Path):
            self.datasets = [self.datasets]
        self.datasets = [pathlib.Path(ds) for ds in self.datasets]
        if self.output_path is None:
            self.output_path = self._generate_output_name()

    def _generate_output_name(self) -> pathlib.Path:
        input_name = pathlib.Path(self.datasets[0])
        return input_name.parent / f"{input_name.parent.name}_mops"

    def feature_info(self) -> dict:
        robocasa_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (9,),
                "names": [
                    "ee_pos_x",
                    "ee_pos_y",
                    "ee_pos_z",
                    "ee_quat_x",
                    "ee_quat_y",
                    "ee_quat_z",
                    "ee_quat_w",
                    "gripper_qpos0",
                    "gripper_qpos1",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (12,),
                "names": [f"robocasa_action_{i}" for i in range(12)],
            },
            # RGB Images
            "observation.images.robot0_agentview_left_image": {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.robot0_agentview_right_image": {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.robot0_eye_in_hand_image": {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
            # Affordance Segmentations
            "observation.images.robot0_agentview_left_segmentation_affordance": {
                "dtype": "lossless",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.robot0_agentview_right_segmentation_affordance": {
                "dtype": "lossless",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.robot0_eye_in_hand_segmentation_affordance": {
                "dtype": "lossless",
                "shape": (self.camera_height, self.camera_width, 3),
                "names": ["height", "width", "channels"],
            },
        }
        return robocasa_features
