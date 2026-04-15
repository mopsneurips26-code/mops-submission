# Adapted from LeRobotDataset to work with MopsIL datasets.

from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import (
    decode_video_frames,
)

from .mops_transforms import MopsImageTransforms
from .video_format import uncompress_mask
from .video_reader import TorchvisionVideoReader


class MopsDataset(LeRobotDataset):
    """Read-only extension of LeRobotDataset for MOPS-IL.

    Differences from the base class:

    - Uses a cached ``TorchvisionVideoReader`` for faster frame decoding.
    - Applies ``MopsImageTransforms`` (geometric + photometric augmentation)
      consistently to paired RGB images and affordance masks.
    - Supports decompressing packed affordance masks (3×uint8 → 24 binary channels).
    - Disables push/pull/download operations (local-only dataset).
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        # Legacy Compatibility: kept for API consistency
        image_transforms: None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> None:
        super().__init__(
            repo_id,
            root,
            episodes,
            None,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync=False,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        # create image transforms with color jitter on contrast, saturation, brightness, hue, and affine transforms
        self.image_transforms = MopsImageTransforms()

        # Track dataset state for efficient incremental writing
        self._lazy_loading = False
        self._recorded_frames = self.meta.total_frames
        self._writer_closed_for_reading = False

    @staticmethod
    def uncompress_batch(batch: dict[str, Any], unc: bool = False) -> dict[str, Any]:
        """Uncompress all affordance masks in the batch."""
        for key in batch:
            if "_segmentation_affordance" in key and "_is_pad" not in key:
                if unc:
                    batch[key] = uncompress_mask(batch[key])
                else:
                    batch[key] = batch[key].float()
        return batch

    def push_to_hub(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Pushing MopsDataset to the Hub is not supported. "
            "Please use LeRobotDataset for this functionality."
        )

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        raise NotImplementedError(
            "Pulling MopsDataset from the Hub is not supported. "
            "Please use LeRobotDataset for this functionality."
        )

    def download(self, download_videos: bool = True) -> None:
        raise NotImplementedError(
            "Downloading MopsDataset from the Hub is not supported. "
            "Please use LeRobotDataset for this functionality."
        )

    def _get_query_indices(
        self, idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int | bool]]]:
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (idx + delta < ep_start) | (idx + delta >= ep_end)
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [
                        self._absolute_to_relative_idx[idx]
                        for idx in query_indices[key]
                    ]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)

            # Fast path for torchvision backends with cached readers
            if self.video_backend in {"pyav", "video_reader"}:
                frames = TorchvisionVideoReader.decode_frames(
                    video_path,
                    shifted_query_ts,
                    self.tolerance_s,
                    backend=self.video_backend,
                )
            else:
                frames = decode_video_frames(
                    video_path, shifted_query_ts, self.tolerance_s, self.video_backend
                )
            item[vid_key] = frames.squeeze(0)

        return item

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                video_frames = self.image_transforms(video_frames)
            item = {**item, **video_frames}

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item

    def __repr__(self) -> str:
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )
