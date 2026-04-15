import dataclasses

import draccus

from lerobot.configs.policies import PreTrainedConfig


@dataclasses.dataclass
class MopsILDefaultConfig(draccus.ChoiceRegistry):
    """Base configuration class for Mops IL default configurations."""

    lr: float | None = None
    input_features: dict | None = None

    enable_state: bool = True
    enable_task: bool = False

    @property
    def name(self) -> str:
        """Name of the configuration."""
        raise NotImplementedError

    def compile(self) -> PreTrainedConfig:
        policy_config = PreTrainedConfig.get_choice_class(self.name)(push_to_hub=False)

        if self.lr is not None:
            policy_config.optimizer_lr = self.lr
            if hasattr(policy_config, "optimizer_lr_backbone"):
                policy_config.optimizer_lr_backbone = self.lr

        if self.input_features is not None:
            policy_config.input_features = self.input_features

        return policy_config


@MopsILDefaultConfig.register_subclass("diffusion")
@dataclasses.dataclass
class DiffusionDefaultConfig(MopsILDefaultConfig):
    """Default configuration for Diffusion-based policies."""

    @property
    def name(self) -> str:
        return "diffusion"

    def compile(self) -> PreTrainedConfig:
        policy_config = super().compile()

        policy_config.crop_shape = (224, 224)
        policy_config.use_separate_rgb_encoder_per_camera = True
        policy_config.down_dims = [128, 256, 512, 512]

        return policy_config


@MopsILDefaultConfig.register_subclass("ditflow")
@dataclasses.dataclass
class DiTFlowDefaultConfig(MopsILDefaultConfig):
    """Default configuration for DiTFlow policies."""

    @property
    def name(self) -> str:
        return "ditflow"

    def compile(self) -> PreTrainedConfig:
        policy_config = super().compile()

        policy_config.crop_shape = (224, 224)
        policy_config.crop_is_random = False

        policy_config.vision_backbone = "resnet34"
        policy_config.pretrained_backbone_weights = "DEFAULT"
        policy_config.use_separate_rgb_encoder_per_camera = True
        policy_config.use_group_norm = policy_config.pretrained_backbone_weights is None

        policy_config.image_only = not self.enable_state
        policy_config.use_text_conditioning = self.enable_task

        policy_config.training_noise_sampling = "beta"
        policy_config.num_blocks = 12

        return policy_config


@MopsILDefaultConfig.register_subclass("affflow")
@dataclasses.dataclass
class AffFlowDefaultConfig(MopsILDefaultConfig):
    """Default configuration for AffFlow policies."""

    @property
    def name(self) -> str:
        return "affflow"

    def compile(self) -> PreTrainedConfig:
        policy_config = super().compile()

        policy_config.crop_shape = (224, 224)
        policy_config.crop_is_random = False

        policy_config.vision_backbone = "resnet34"
        policy_config.pretrained_backbone_weights = "DEFAULT"
        policy_config.use_separate_rgb_encoder_per_camera = True
        policy_config.use_group_norm = policy_config.pretrained_backbone_weights is None

        policy_config.image_only = not self.enable_state
        policy_config.use_text_conditioning = self.enable_task

        policy_config.training_noise_sampling = "beta"
        policy_config.num_blocks = 12

        return policy_config


@MopsILDefaultConfig.register_subclass("mopsflow")
@dataclasses.dataclass
class MopsFlowDefaultConfig(MopsILDefaultConfig):
    """Default configuration for MopsFlow policies."""

    @property
    def name(self) -> str:
        return "mopsflow"

    def compile(self) -> PreTrainedConfig:
        policy_config = super().compile()

        policy_config.crop_shape = (224, 224)
        policy_config.crop_is_random = False

        policy_config.vision_backbone = "resnet34"
        policy_config.pretrained_backbone_weights = "DEFAULT"
        policy_config.use_separate_rgb_encoder_per_camera = True
        policy_config.use_group_norm = policy_config.pretrained_backbone_weights is None

        policy_config.image_only = not self.enable_state
        policy_config.use_text_conditioning = self.enable_task

        policy_config.training_noise_sampling = "beta"
        policy_config.num_blocks = 12

        policy_config.segmentation_prediction_loss_weight = 0.1
        policy_config.affordance_keys = [
            "observation.images.robot0_agentview_left_segmentation_affordance",
            "observation.images.robot0_agentview_right_segmentation_affordance",
            "observation.images.robot0_eye_in_hand_segmentation_affordance",
        ]

        return policy_config
