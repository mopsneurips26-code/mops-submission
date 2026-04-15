"""Environment utilities for evaluation."""

import numpy as np
from loguru import logger

import robosuite
from mops_il.data_ops.video_format import extract_images, extract_state
from mopscasa.env_utils import load_env_info
from mopscasa.observation_extender import ObservationExtender
from robocasa.environments.kitchen.kitchen import Kitchen


def create_env(
    env_name: str, seed: int = 42, img_size: int = 256, show: bool = False
) -> Kitchen:
    """Create a Robocasa environment.

    Args:
        env_name: Name of the environment.
        seed: Random seed.
        img_size: Image size for rendering.
        show: Whether to show the environment rendering.

    Returns:
        The created Kitchen environment.
    """
    all_env_kwargs = load_env_info()

    logger.info(f"Creating environment: {env_name} with seed {seed}")
    env_kwargs = all_env_kwargs[env_name]
    env_kwargs["env_name"] = env_name
    env_kwargs["seed"] = seed
    env_kwargs["camera_heights"] = img_size
    env_kwargs["camera_widths"] = img_size
    env_kwargs["camera_segmentations"] = "element"
    if show:
        env_kwargs["has_renderer"] = True
        env_kwargs["has_offscreen_renderer"] = True

    env: Kitchen = robosuite.make(**env_kwargs)
    return env


def prepare_observation(
    obs: dict, obs_extender: ObservationExtender, img_size: int
) -> dict:
    """Prepare observation for the policy.

    Args:
        obs: Raw observation from the environment.
        obs_extender: Observation extender instance.
        img_size: Image size for resizing.

    Returns:
        Processed observation dictionary.
    """
    obs = obs_extender.augment_observation(obs, flip_vert=False)
    imgs = extract_images(obs, -1, img_size, img_size)

    # Vertically Flip images to match dataset convention
    for key in imgs:
        imgs[key] = np.flipud(imgs[key])

    state = extract_state(obs, -1)
    obs_out = imgs | state
    return obs_out
