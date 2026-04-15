import datetime as dt
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs.train import (
    DatasetConfig,
    PreTrainedConfig,
    TrainPipelineConfig,
    WandBConfig,
)
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from mops_il.config.default_configs import MopsILDefaultConfig


@dataclass
class MopsConfigCLI:
    # Path to Lerobot Dataset
    dataset_path: Path
    # Policy type
    policy: str = "ditflow"
    # Whether to enable state as input to the policy
    enable_state: bool = True
    # Whether to enable RGB images as input to the policy
    enable_rgb: bool = True
    # Whether to enable affordances masks as input to the policy
    enable_affordances: bool = True
    # Whether to enable depth as input to the policy
    enable_depth: bool = False
    # Whether to enable task conditioning
    enable_task: bool = False
    # Whether to uncompress affordance masks when loading dataset
    uncompress_mask: bool = False
    # Batch size for training
    batch_size: int = 32
    # Learning rate. None to use default from optimizer config
    lr: float | None = None
    # Total training steps
    steps: int = 100_000
    # Dataloader workers
    num_workers: int = 4
    # Disable WandB logging.
    disable_wandb: bool = False

    def create_config(self) -> TrainPipelineConfig:
        """Build a complete TrainPipelineConfig from CLI arguments.

        Assembles dataset, policy, and WandB configs, resolves input features
        from the dataset metadata, and generates a descriptive job name.

        Returns:
            A fully initialized TrainPipelineConfig ready for training.
        """
        dataset_config = self.create_dataset_config_from_cli()
        policy_config = self.create_policy_config_from_cli()

        policy_config.input_features = self.build_input_features()

        wandb_config = self.create_wandb_config_from_cli()
        job_name = self._build_job_name()
        cfg = TrainPipelineConfig(
            job_name=job_name,
            dataset=dataset_config,
            policy=policy_config,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            steps=self.steps,
            save_freq=min(self.steps // 10, 25_000),
            wandb=wandb_config,
        )
        return cfg

    def create_policy_config_from_cli(
        self,
    ) -> PreTrainedConfig:
        """Create a policy config from CLI flags.

        Looks up the default config for the selected policy type, applies
        the selected input modalities, and compiles into a PreTrainedConfig.

        Returns:
            Compiled policy configuration.
        """
        selected_inputs = self._select_policy_inputs()
        default_config = MopsILDefaultConfig.get_choice_class(self.policy)(
            lr=self.lr,
            input_features=selected_inputs,
            enable_state=self.enable_state,
            enable_task=self.enable_task,
        )
        return default_config.compile()

    def create_dataset_config_from_cli(
        self,
    ) -> DatasetConfig:
        """Create a DatasetConfig pointing to the local dataset path.

        Returns:
            Dataset configuration with repo_id and root set.
        """
        ds = DatasetConfig(
            repo_id=str(self.dataset_path),
            root=self.dataset_path,
        )
        ds.uncompress_mask = self.uncompress_mask

        return ds

    def create_wandb_config_from_cli(
        self,
    ) -> WandBConfig:
        """Create a WandB config from CLI flags.

        Returns:
            WandB configuration with project name and enable flag.
        """
        return WandBConfig(
            enable=not self.disable_wandb, project="mops-il", disable_artifact=True
        )

    def _build_job_name(self) -> str:
        """Generate a descriptive job name from the current config.

        Format: ``MMDDHHMM_<policy>_[d]_[aff]_[norgb]_[nostate]``.

        Returns:
            A short, human-readable job name string.
        """
        parts = [dt.datetime.now().strftime("%m%d%H%M"), self.policy[:3]]

        if self.enable_depth:
            parts.append("d")
        if self.enable_affordances:
            parts.append("aff")

        if not self.enable_rgb:
            parts.append("norgb")

        if not self.enable_state:
            parts.append("nostate")

        return "_".join(parts)

    def _select_policy_inputs(self) -> list[str]:
        """Return observation keys that are enabled via CLI flags.

        Returns:
            List of dataset feature keys to use as policy inputs.
        """
        selected_inputs = []
        if self.enable_state:
            selected_inputs.append("observation.state")
        if self.enable_rgb:
            selected_inputs.append("observation.images.robot0_agentview_left_image")
            selected_inputs.append("observation.images.robot0_agentview_right_image")
            selected_inputs.append("observation.images.robot0_eye_in_hand_image")
        if self.enable_affordances:
            selected_inputs.append(
                "observation.images.robot0_agentview_left_segmentation_affordance"
            )
            selected_inputs.append(
                "observation.images.robot0_agentview_right_segmentation_affordance"
            )
            selected_inputs.append(
                "observation.images.robot0_eye_in_hand_segmentation_affordance"
            )
        if self.enable_depth:
            selected_inputs.append("observation.images.robot0_agentview_left_depth")
            selected_inputs.append("observation.images.robot0_agentview_right_depth")
            selected_inputs.append("observation.images.robot0_eye_in_hand_depth")
        return selected_inputs

    def build_input_features(self) -> dict:
        """Build the input feature dict from dataset metadata and CLI flags.

        Loads the dataset feature schema, filters to only enabled modalities,
        and adjusts affordance mask shapes when uncompression is enabled
        (3 channels → 24 binary channels).

        Returns:
            Dict mapping feature keys to their PolicyFeature descriptors.
        """
        meta_data = LeRobotDatasetMetadata(
            repo_id=str(self.dataset_path),
            root=self.dataset_path,
        )
        dataset_features = dataset_to_policy_features(meta_data.features)
        selected_features = self._select_policy_inputs()

        all_features = deepcopy(dataset_features)
        input_features = {
            k: v for k, v in all_features.items() if k in selected_features
        }

        if self.uncompress_mask:
            # Update affordance mask shapes if uncompressing
            for key in input_features:
                if "affordance" in key:
                    input_features[key].shape = (
                        # MOPS-IL Dataset affordance masks have 24 channels after uncompressing
                        24,
                        input_features[key].shape[1],
                        input_features[key].shape[2],
                    )

        return input_features
