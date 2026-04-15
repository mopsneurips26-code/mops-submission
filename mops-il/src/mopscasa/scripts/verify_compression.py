import dataclasses
import pathlib

import draccus
import h5py
import torch
from loguru import logger

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass
class VerifyCompressionConfig:
    lr_path: pathlib.Path
    h5_path: pathlib.Path
    frame: int = 0


def decompress_torch_video(tensor: torch.Tensor) -> torch.Tensor:
    # aff_image is 3HW float image.
    # Times 255 and converted to uint8 for storage.
    aff_img_uint8 = tensor  # * 255).to(torch.uint8)
    logger.info(f"Decompressed affordance image shape: {aff_img_uint8.shape}")

    # Interpret the uint8 image as bitmask and reshape to 24 channels
    aff_img_bitmask = torch.zeros(
        (24, aff_img_uint8.shape[1], aff_img_uint8.shape[2]), dtype=torch.bool
    )
    for bit in range(8):
        aff_img_bitmask[bit, :, :] = (aff_img_uint8[0, :, :] & (1 << bit)) != 0
        aff_img_bitmask[bit + 8, :, :] = (aff_img_uint8[1, :, :] & (1 << bit)) != 0
        aff_img_bitmask[bit + 16, :, :] = (aff_img_uint8[2, :, :] & (1 << bit)) != 0
    return aff_img_bitmask


def decompress_torch_video_final_fixed_bit_order(tensor: torch.Tensor) -> torch.Tensor:
    # Input tensor is C=3, H, W (R, G, B order, assumed correct after rgb24)
    aff_img_uint8 = tensor
    aff_img_bitmask = torch.zeros(
        (24, aff_img_uint8.shape[1], aff_img_uint8.shape[2]), dtype=torch.bool
    )

    # We must match the ground truth: Seg. Channel i should be mapped to the i-th bit of the 24-bit value.
    for i in range(24):
        # Calculate which of the 3 bytes this channel (i) belongs to
        byte_index = i // 8  # 0, 1, or 2 (R, G, B)

        # Calculate the bit position (j) within that byte (0 to 7)
        bit_pos_in_byte = i % 8

        # We need to map the segmentation channel i to the correct bit (1 << bit_pos_in_byte)
        # We try the original logic again, but confirm the channels are R, G, B (0, 1, 2)

        # Try the LSB-first assumption again (which gave 0.28)
        is_set = (aff_img_uint8[byte_index, :, :] & (1 << bit_pos_in_byte)) != 0

        aff_img_bitmask[i, :, :] = is_set

    return aff_img_bitmask


def decompress_uint32_video(tensor: torch.Tensor) -> torch.Tensor:
    # aff_image is 1HW uint32 image.
    aff_img_uint32 = tensor.to(torch.uint32)
    logger.info(f"Decompressed affordance image shape: {aff_img_uint32.shape}")

    # Interpret the uint32 image as bitmask and reshape to 32 channels
    aff_img_bitmask = torch.zeros(
        (24, aff_img_uint32.shape[1], aff_img_uint32.shape[2]), dtype=torch.bool
    )
    for bit in range(24):
        aff_img_bitmask[bit, :, :] = (aff_img_uint32[0, :, :] & (1 << bit)) != 0
    return aff_img_bitmask


def get_h5_sample(h5_path: pathlib.Path, frame: int = 0) -> torch.Tensor:
    logger.info(f"Loading H5 sample from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        d = f["data"]["demo_0"]
        obs = d["obs"]
        logger.info(f"Available observation keys: {list(obs.keys())}")
        logger.info(obs["robot0_agentview_right_segmentation_affordance"])
        h5_sample = torch.from_numpy(
            obs["robot0_agentview_right_segmentation_affordance"][frame]
        )

    # swap axes to CHW
    h5_sample = h5_sample.permute(2, 0, 1)
    h5_sample = decompress_uint32_video(h5_sample)
    return h5_sample


def get_lerobot_sample(lr_path: pathlib.Path, frame: int = 0) -> torch.Tensor:
    ds = LeRobotDataset(
        repo_id=str(lr_path),
        root=lr_path,
    )

    verification_key = (
        "observation.images.robot0_agentview_right_segmentation_affordance"
    )

    batch_data = ds[0]
    aff_tensor = batch_data[verification_key]
    aff_img_bitmask = decompress_torch_video(aff_tensor)
    return aff_img_bitmask


def verify_compression(config: VerifyCompressionConfig) -> None:
    h5_sample = get_h5_sample(config.h5_path, config.frame)
    lerobot_sample = get_lerobot_sample(config.lr_path, config.frame)

    logger.info(f"H5 sample shape: {h5_sample.shape}, dtype: {h5_sample.dtype}")
    logger.info(
        f"LeRobot sample shape: {lerobot_sample.shape}, dtype: {lerobot_sample.dtype}"
    )

    # Compare the two samples
    if torch.equal(h5_sample, lerobot_sample):
        logger.info("Verification successful: The samples match!")
    else:
        logger.error("Verification failed: The samples do not match.")

    # Compute IOU for each channel independently
    ious = []
    for channel in range(h5_sample.shape[0]):
        h5_channel = h5_sample[channel]
        lr_channel = lerobot_sample[channel]

        intersection = torch.logical_and(h5_channel, lr_channel).sum().item()
        union = torch.logical_or(h5_channel, lr_channel).sum().item()

        iou = intersection / union if union > 0 else 1.0
        ious.append(iou)
        logger.info(f"Channel {channel}: IOU = {iou:.4f}")

    # Total IOU
    total_intersection = torch.logical_and(h5_sample, lerobot_sample).sum().item()
    total_union = torch.logical_or(h5_sample, lerobot_sample).sum().item()
    total_iou = total_intersection / total_union if total_union > 0 else 1.0
    logger.info(f"Total IOU across all channels: {total_iou:.4f}")


if __name__ == "__main__":
    config = draccus.parse(VerifyCompressionConfig)

    # Placeholder for actual verification logic
    logger.info(f"Verifying compression for dataset at: {config.lr_path}")
    verify_compression(config)
