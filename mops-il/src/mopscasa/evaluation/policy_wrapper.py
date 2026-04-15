"""Policy wrapper for evaluation."""

from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


class FullPolicyWrapper:
    """Wrapper for the policy that includes pre- and post-processing."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        pre_processor: PolicyProcessorPipeline,
        post_processor: PolicyProcessorPipeline,
        uncompress_aff: bool = False,
    ) -> None:
        """Initialize the policy wrapper.

        Args:
            policy: The raw policy model.
            pre_processor: The pre-processor for observations.
            post_processor: The post-processor for actions.
        """
        self.policy = policy
        self.pre_processor = pre_processor
        self.post_processor = post_processor

        self.uncompress_aff = uncompress_aff

    def select_action(self, observations: dict[str, Any]) -> PolicyAction:
        """Select an action given observations.

        Args:
            observations: The observation dictionary.

        Returns:
            The selected action.
        """
        processed_obs = self.pre_processor(observations)
        with torch.no_grad():
            raw_actions = self.policy.select_action(processed_obs)
        actions = self.post_processor(raw_actions)
        return actions

    def reset(self) -> None:
        """Reset the policy state."""
        self.policy.reset()
