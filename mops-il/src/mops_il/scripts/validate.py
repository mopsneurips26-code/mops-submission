#!/usr/bin/env python
"""Visualization script comparing policy predictions to ground-truth actions.

Generates per-episode plots showing each action dimension over time, overlaying
the policy prediction (red) on top of the ground-truth trajectory (black).

Usage::

    python src/mops_il/scripts/validate.py --checkpoint <path> --episodes 0 1 2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from mops_il.scripts.evaluate import get_pretrained_model_path, load_policy
from mops_il.train import make_dataset


def validate(
    checkpoint_path: Path, episode_indices: list[int], output_dir: Path
) -> None:
    """Run validation by comparing model predictions to ground-truth actions.

    For each requested episode, iterates through all frames, runs the policy
    in inference mode, and saves a matplotlib figure with per-dimension
    comparison plots.

    Args:
        checkpoint_path: Path to the training checkpoint directory.
        episode_indices: Episode indices to visualize.
        output_dir: Directory to save the output plots.
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        cfg_path = get_pretrained_model_path(checkpoint_path)
        cfg = TrainPipelineConfig.from_pretrained(cfg_path)
    except Exception as e:
        print(f"Failed to load TrainPipelineConfig: {e}")
        return

    # Create Dataset
    print(f"Loading dataset: {cfg.dataset.repo_id}")
    dataset = make_dataset(cfg)

    # Load Policy
    print(f"Loading policy from {checkpoint_path}")
    policy, policy_device = load_policy(checkpoint_path)

    episodes_meta = dataset.meta.episodes
    # Access meta safely
    if hasattr(episodes_meta, "keys"):  # Dict-like
        num_episodes = len(episodes_meta["dataset_from_index"])
    else:  # List-like (legacy?)
        num_episodes = len(episodes_meta)

    for ep_idx in episode_indices:
        print(f"Processing Episode {ep_idx}...")

        if ep_idx >= num_episodes:
            print(f"Episode {ep_idx} out of range (max {num_episodes - 1})")
            continue

        # Get start/end indices
        if hasattr(episodes_meta, "keys"):
            start_idx = int(episodes_meta["dataset_from_index"][ep_idx])
            end_idx = int(episodes_meta["dataset_to_index"][ep_idx])
        else:
            start_idx = int(episodes_meta[ep_idx]["dataset_from_index"])
            end_idx = int(episodes_meta[ep_idx]["dataset_to_index"])

        preds = []
        gts = []

        # Iterate over frames in the episode
        for i in range(start_idx, end_idx):
            item = dataset[i]

            action_pred = policy.select_action(item)

            action_gt = item["action"]

            pred_np = action_pred.detach().cpu().numpy()

            # Handle batch dimension if present in output
            if pred_np.ndim == 2 and pred_np.shape[0] == 1:
                pred_np = pred_np[0]

            preds.append(pred_np)
            gts.append(action_gt.numpy())

        preds = np.array(preds)
        gts = np.array(gts)

        print(f"Predictions shape: {preds.shape}, Ground Truth shape: {gts.shape}")

        # Plot
        dim = preds.shape[-1]
        fig, axes = plt.subplots(dim, 1, figsize=(10, 2 * dim), sharex=True)
        if dim == 1:
            axes = [axes]

        for d in range(dim):
            axes[d].plot(gts[:, 0, d], label="Ground Truth", color="black", alpha=0.7)
            axes[d].plot(preds[:, 0, d], label="Prediction", color="red", alpha=0.7)
            axes[d].set_ylabel(f"Dim {d}")
            if d == 0:
                axes[d].legend()

        plt.xlabel("Step")
        plt.suptitle(f"Episode {ep_idx} - Prediction vs Ground Truth")
        plt.tight_layout()

        save_path = output_dir / f"episode_{ep_idx}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_output",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    validate(Path(args.checkpoint), args.episodes, Path(args.output))
