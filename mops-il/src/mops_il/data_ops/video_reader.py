from pathlib import Path
from typing import Any

import torch
import torchvision


class TorchvisionVideoReader:
    """Lightweight, process-local cache for torchvision VideoReader instances to avoid
    reopening the same file on every sample. Each DataLoader worker is a separate
    process, so this stays isolated per worker.
    """

    _READERS: dict[str, Any] = {}

    @classmethod
    def _get_reader(cls, path: str, backend: str | None) -> Any:
        if path in cls._READERS:
            return cls._READERS[path]

        if backend is not None:
            torchvision.set_video_backend(backend)

        reader = torchvision.io.VideoReader(path, "video")
        cls._READERS[path] = reader
        return reader

    @classmethod
    def decode_frames(
        cls,
        video_path: Path | str,
        timestamps: list[float],
        tolerance_s: float,
        backend: str = "pyav",
    ) -> torch.Tensor:
        """Decode frames using a cached torchvision VideoReader for speed.

        Mirrors the behavior of lerobot.datasets.video_utils.decode_video_frames_torchvision
        but keeps the reader open across calls to avoid re-opening overhead.
        """
        # Ensure sorted timestamps for efficient linear read
        first_ts = min(timestamps)
        last_ts = max(timestamps)

        reader = cls._get_reader(str(video_path), backend)

        # pyav backend only supports keyframe seeks
        keyframes_only = backend == "pyav"
        reader.seek(first_ts, keyframes_only=keyframes_only)

        loaded_frames = []
        loaded_ts = []
        for frame in reader:
            current_ts = frame["pts"]
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        query_ts = torch.tensor(timestamps)
        loaded_ts_t = torch.tensor(loaded_ts)

        # compute distances between each query timestamp and timestamps of all loaded frames
        dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
        min_, argmin_ = dist.min(1)

        is_within_tol = min_ < tolerance_s
        if not is_within_tol.all():
            raise AssertionError(
                "Requested frames violate tolerance. "
                f"max delta={float(min_[~is_within_tol].max())} > tolerance_s={tolerance_s}"
            )

        closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])

        # Match lerobot decode behavior: uint8, channel-first
        closest_frames = closest_frames.type(torch.uint8)

        return closest_frames
