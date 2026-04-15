#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .core import EnvTransition, TransitionKey
from .pipeline import (
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # NOTE (maractingi): No mps kernel for crop and resize, so we need to move to cpu
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """Tracks episode steps and enforces a time limit by truncating the episode.

    Attributes:
        max_episode_steps: The maximum number of steps allowed per episode.
        current_step: The current step count for the active episode.
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """Increments the step counter and sets the truncated flag if the time limit is reached.

        Args:
            truncated: The incoming truncated flag.

        Returns:
            True if the episode step limit is reached, otherwise the incoming value.
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): missing an else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the `max_episode_steps`.
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """Resets the step counter, typically called at the start of a new episode."""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ComplementaryDataProcessorStep):
    """Applies a penalty for inefficient gripper usage.

    This step penalizes actions that attempt to close an already closed gripper or
    open an already open one, based on position thresholds.

    Attributes:
        penalty: The negative reward value to apply.
        max_gripper_pos: The maximum position value for the gripper, used for normalization.
    """

    penalty: float = -0.01
    max_gripper_pos: float = 30.0

    def complementary_data(self, complementary_data: dict) -> dict:
        """Calculates the gripper penalty and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data, which should contain
                                raw joint positions.

        Returns:
            A new complementary data dictionary with the `discrete_penalty` key added.
        """
        action = self.transition.get(TransitionKey.ACTION)

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return complementary_data

        current_gripper_pos = raw_joint_positions.get(GRIPPER_KEY, None)
        if current_gripper_pos is None:
            return complementary_data

        # Gripper action is a PolicyAction at this stage
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # Calculate penalty boolean as in original
        gripper_penalty_bool = (
            gripper_state_normalized < 0.5 and gripper_action_normalized > 0.5
        ) or (gripper_state_normalized > 0.75 and gripper_action_normalized < 0.5)

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Create new complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty

        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the penalty value and max gripper position.
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
        }

    def reset(self) -> None:
        """Resets the processor's internal state."""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """Applies a pretrained reward classifier to image observations to predict success.

    This step uses a model to determine if the current state is successful, updating
    the reward and potentially terminating the episode.

    Attributes:
        pretrained_path: Path to the pretrained reward classifier model.
        device: The device to run the classifier on.
        success_threshold: The probability threshold to consider a prediction as successful.
        success_reward: The reward value to assign on success.
        terminate_on_success: If True, terminates the episode upon successful classification.
        reward_classifier: The loaded classifier model instance.
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """Initializes the reward classifier model after the dataclass is created."""
        if self.pretrained_path is not None:
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Processes a transition, applying the reward classifier to its image observations.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with an updated reward and done flag based on the
            classifier's prediction.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(
                images, threshold=self.success_threshold
            )

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # Calculate reward and termination
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
