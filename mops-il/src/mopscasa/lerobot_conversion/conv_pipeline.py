import json
import pathlib

import h5py
import numpy as np
import tqdm
from loguru import logger

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mops_il.data_ops.video_format import extract_images, extract_state

from .conv_config import ConversionConfig


def create_lerobot_dataset(
    cfg: ConversionConfig, n_img_writer_proc: int = 8
) -> LeRobotDataset:
    cfg = cfg
    lerobot_dataset = LeRobotDataset.create(
        repo_id="local_only",
        features=cfg.feature_info(),
        fps=30,
        root=cfg.output_path,
        use_videos=True,
        image_writer_threads=n_img_writer_proc * 2,
        image_writer_processes=n_img_writer_proc,
    )

    logger.info(f"Creating LeRobot dataset at {cfg.output_path}")
    total_num_ds = len(cfg.datasets)
    for i, ds in enumerate(cfg.datasets):
        logger.info(f"Processing dataset: {ds.name} ({i + 1}/{total_num_ds})")
        traverse_h5_file(
            ds, lerobot_dataset, cfg.camera_height, cfg.camera_width, i, total_num_ds
        )
    logger.info(f"Finished creating LeRobot dataset {cfg.output_path}")


def traverse_h5_file(
    ds_path: pathlib.Path,
    lerobot_dataset: LeRobotDataset,
    h: int,
    w: int,
    ds_index: int = 0,
    total_ds: int = 1,
) -> None:
    with h5py.File(ds_path, "r") as f:
        data = f["data"]
        total_demos = len(data)
        for i, demo in enumerate(data):
            logger.info(
                f"DS: {ds_index + 1}/{total_ds} {ds_path.parent.name} - DEMO: {demo} {i + 1}/{total_demos} - START"
            )
            process_demo(data[demo], lerobot_dataset, h, w, ds_path)
            logger.info(
                f"DS: {ds_index + 1}/{total_ds} {ds_path.parent.name} - DEMO: {demo} {i + 1}/{total_demos} - END"
            )


def process_demo(
    traj: h5py.Group,
    lerobot_dataset: LeRobotDataset,
    h: int,
    w: int,
    ds_path: pathlib.Path,
) -> None:
    task = json.loads(traj.attrs["ep_meta"])["lang"]
    num_frames = traj["actions"].shape[0]

    for i in tqdm.trange(num_frames):
        frame = {"task": task}
        frame["action"] = traj["actions"][i].astype(np.float32)
        frame.update(extract_state(traj["obs"], i))
        frame.update(extract_images(traj["obs"], i, h, w))
        lerobot_dataset.add_frame(frame)

    episode_index = lerobot_dataset.meta.total_episodes
    lerobot_dataset.save_episode()

    # Write mapping
    map_file = lerobot_dataset.root / "episode_map.jsonl"
    with open(map_file, "a") as f:
        json.dump(
            {"episode_index": episode_index, "h5_file": str(ds_path.parent.name)}, f
        )
        f.write("\n")
