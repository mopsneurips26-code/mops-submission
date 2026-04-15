"""Evaluation script for MopsCasa policies.

This script loads a trained policy and evaluates it on a set of environments.
"""

import dataclasses
import pathlib

import draccus
import torch
from loguru import logger

import wandb
from lerobot.configs.policies import PreTrainedConfig
from mops_il.policies.factory import get_policy_class, make_pre_post_processors
from mopscasa.env_utils import load_env_info
from mopscasa.evaluation.metrics import EvaluationMetrics
from mopscasa.evaluation.policy_wrapper import FullPolicyWrapper
from mopscasa.evaluation.runner import run_parallel_evaluation


@dataclasses.dataclass
class EvalCLI:
    """Command line arguments for evaluation."""

    # Checkpoint Path
    checkpoint: pathlib.Path
    # Show Simulation onscreen
    show: bool = False
    # Num of parallel workers
    num_workers: int = 4
    # Num of trials per environment
    max_trials: int = 10
    # Max steps per trial
    max_steps: int = 500
    # Image size for observation
    img_size: int = 256
    # Model Random Seed
    seed: int = 1000

    # WandB arguments
    wandb_project: str = "mopscasa-eval"
    wandb_job_type: str = "eval"


def get_pretrained_model_path(path: pathlib.Path) -> pathlib.Path:
    """Find the pretrained model path within a checkpoint directory.

    Args:
        path: Path to the checkpoint directory.

    Returns:
        Path to the pretrained model directory.

    Raises:
        FileNotFoundError: If the pretrained model cannot be found.
    """
    if (path / "config.json").exists():
        return path

    # load latest model checkpoint:
    checkpoints_dirs = sorted((path / "checkpoints").glob("*/"), reverse=True)
    latest_checkpoint = checkpoints_dirs[0] / "pretrained_model"
    if latest_checkpoint.exists():
        return latest_checkpoint

    raise FileNotFoundError(
        f"Could not find pretrained model in {path} or {latest_checkpoint}"
    )


def load_policy(
    checkpoint_path: pathlib.Path, seed: int
) -> tuple[FullPolicyWrapper, str]:
    """Load a policy from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint.

    Returns:
        A tuple containing the loaded policy wrapper and the device (cpu or cuda).
    """
    pretrained_path = get_pretrained_model_path(checkpoint_path)
    logger.info(f"Loading policy from {pretrained_path}")

    # Load config to determine policy type
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    cfg.seed = seed

    logger.info(f"Detected policy type: {cfg.type}")

    PolicyCls = get_policy_class(cfg.type)
    policy = PolicyCls.from_pretrained(pretrained_path)
    policy.eval()

    device = "cpu"
    if torch.cuda.is_available():
        policy.cuda()
        logger.info("Moved policy to CUDA")
        device = "cuda"

    # Read shape of visual inputs to determine Affordance uncompression
    uncompress_aff = False
    for key, feature in cfg.image_features.items():
        if "_segmentation_affordance" in key and feature.shape[0] == 24:
            uncompress_aff = True
            break

    pre_processor, post_processor = make_pre_post_processors(cfg, str(pretrained_path))
    policy = FullPolicyWrapper(
        policy, pre_processor, post_processor, uncompress_aff=uncompress_aff
    )

    return policy, device


def eval_all_envs(
    policy: FullPolicyWrapper,
    device: str,
    num_workers: int,
    max_trials: int = 5,
    max_steps: int = 1000,
    show: bool = False,
    img_size: int = 256,
) -> None:
    """Evaluate the policy on all environments.

    Args:
        policy: The policy to evaluate.
        device: Device to run the policy on.
        num_workers: Number of parallel workers.
        max_trials: Number of trials per environment.
        show: Whether to render the environment.
        img_size: Image size for observation.
    """
    env_info = load_env_info()
    all_envs = list(env_info.keys())

    metrics = EvaluationMetrics(use_wandb=wandb.run is not None)

    # Create a flat list of all tasks
    tasks = []
    for env_idx, env_name in enumerate(all_envs):
        for i in range(max_trials):
            tasks.append(
                {
                    "env_name": env_name,
                    "env_idx": env_idx,
                    "seed": i,  # Use trial index as seed
                }
            )

    logger.info(f"Generated {len(tasks)} tasks across {len(all_envs)} environments")

    results = run_parallel_evaluation(
        tasks=tasks,
        policy=policy,
        device=device,
        num_workers=num_workers,
        show=show,
        max_steps=max_steps,
        img_size=img_size,
    )

    # Record results
    for env_name, success in results:
        metrics.record_trial(env_name, success)

    metrics.log_summary()


def main(cli_args: EvalCLI) -> None:
    """Main entry point for evaluation.

    Args:
        cli_args: Command line arguments.
    """
    wandb.init(
        project=cli_args.wandb_project,
        job_type=cli_args.wandb_job_type,
        config=dataclasses.asdict(cli_args),
    )

    policy, device = load_policy(cli_args.checkpoint, cli_args.seed)
    logger.info("Policy loaded successfully")
    eval_all_envs(
        policy,
        device,
        num_workers=cli_args.num_workers,
        max_trials=cli_args.max_trials,
        max_steps=cli_args.max_steps,
        show=cli_args.show,
        img_size=cli_args.img_size,
    )

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    args = draccus.parse(config_class=EvalCLI)
    main(args)
