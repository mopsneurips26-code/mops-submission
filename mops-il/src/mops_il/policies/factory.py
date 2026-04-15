from collections.abc import Callable
from typing import Any, TypeVar

import torch
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import ProcessorConfigKwargs
from lerobot.policies.factory import get_policy_class as lerobot_get_policy_class
from lerobot.policies.factory import (
    make_pre_post_processors as lerobot_make_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

PolicyCls = TypeVar("PolicyCls", bound=type[PreTrainedPolicy])
ProcessorFn = TypeVar("ProcessorFn", bound=Callable[..., Any])


_POLICY_REGISTRY: dict[str, type[PreTrainedPolicy]] = {}
_PROCESSOR_FN_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_policy(
    policy_type: str, policy_cls: type[PreTrainedPolicy] | None = None
) -> Callable[[PolicyCls], PolicyCls] | PolicyCls:
    """Register a policy class; usable as a decorator or direct call."""

    def decorator(cls: PolicyCls) -> PolicyCls:
        _POLICY_REGISTRY[policy_type] = cls
        return cls

    if policy_cls is None:
        return decorator
    return decorator(policy_cls)


def get_policy_class(policy_type: str) -> type[PreTrainedPolicy]:
    if policy_type in _POLICY_REGISTRY:
        return _POLICY_REGISTRY[policy_type]
    return lerobot_get_policy_class(policy_type)


def register_processor_fn(
    policy_type: str, processor_fn: Callable[..., Any] | None = None
) -> Callable[[ProcessorFn], ProcessorFn] | ProcessorFn:
    """Register a pre/post-processor factory; decorator or direct call."""

    def decorator(fn: ProcessorFn) -> ProcessorFn:
        _PROCESSOR_FN_REGISTRY[policy_type] = fn
        return fn

    if processor_fn is None:
        return decorator
    return decorator(processor_fn)


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if policy_cfg.type in _PROCESSOR_FN_REGISTRY:
        return _PROCESSOR_FN_REGISTRY[policy_cfg.type](
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    return lerobot_make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=pretrained_path,
        **kwargs,
    )


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg=None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Condensed version from lerobot.policies.factory.make_policy

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError(
            "Either one of a dataset metadata or a sim env must be provided."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is None:
        raise ValueError("Dataset metadata cannot be none.")
    features = dataset_to_policy_features(ds_meta.features)

    if not cfg.output_features:
        cfg.output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
        }
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
    return policy
