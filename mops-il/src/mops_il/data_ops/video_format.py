"""Shared helpers for MOPS custom video/observation formats."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

CAM_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def extract_state(obs: dict[str, Any], idx: int) -> dict[str, np.ndarray]:
    """Extract state vector from a raw observation dict.

    Args:
        obs: Observation dictionary with state fields.
        idx: Frame index to extract when a time dimension is present.

    Returns:
        Dict with a single key containing the concatenated state vector.
    """
    eef_pos = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    gripper_qpos = obs["robot0_gripper_qpos"]

    if eef_pos.ndim == 2:
        eef_pos = eef_pos[idx]
        eef_quat = eef_quat[idx]
        gripper_qpos = gripper_qpos[idx]

    return {
        "observation.state": np.concatenate([eef_pos, eef_quat, gripper_qpos]).astype(
            np.float32
        )
    }


def extract_images(
    obs: dict[str, Any],
    idx: int,
    h: int,
    w: int,
    cam_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract and resize image + affordance channels into LeRobot keys."""
    images: dict[str, np.ndarray] = {}
    cams = cam_names or CAM_NAMES

    for cam_name in cams:
        img_key = f"observation.images.{cam_name}_image"
        images[img_key] = _extract_single_image(obs, f"{cam_name}_image", idx, h, w)

        afford_key = f"observation.images.{cam_name}_segmentation_affordance"
        images[afford_key] = _extract_single_image(
            obs, f"{cam_name}_segmentation_affordance", idx, h, w
        )

    return images


def _extract_single_image(
    obs: dict[str, Any], key: str, idx: int, h: int, w: int
) -> np.ndarray:
    img = obs[key]
    if img.ndim == 4:
        img = img[idx]
    return proc_image(h, w, img)


def proc_image(h: int, w: int, img: np.ndarray) -> np.ndarray:
    """Process image before saving to LeRobot dataset.

    Args:
        h: Target height.
        w: Target width.
        img: Input image array.

    Returns:
        Processed image.
    """
    new_height = h
    new_width = w
    if img.shape[0] != new_height or img.shape[1] != new_width:
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    if img.dtype == np.uint32:
        r = (img & 0x000000FF).astype(np.uint8)
        g = ((img & 0x0000FF00) >> 8).astype(np.uint8)
        b = ((img & 0x00FF0000) >> 16).astype(np.uint8)
        img = np.stack([r, g, b], axis=-1)
        img = img.squeeze()

    return img


def uncompress_mask(batch: torch.Tensor) -> torch.Tensor:
    """Uncompress affordance mask from uint8 to original labels.

    Args:
        batch: Compressed affordance mask tensor (..., C, H, W).

    Returns:
        Uncompressed affordance mask tensor (..., C*8, H, W).
    """
    *batch_dims, channels, height, width = batch.shape

    bits = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], device=batch.device, dtype=torch.uint8
    )

    mask_flat = batch.view(-1, channels, height, width).to(torch.uint8)
    unpacked = (mask_flat.unsqueeze(2) & bits.view(1, 1, 8, 1, 1)) > 0
    aff_img_bitmask = unpacked.reshape(-1, channels * 8, height, width).float()

    return aff_img_bitmask.view(*batch_dims, -1, height, width)
