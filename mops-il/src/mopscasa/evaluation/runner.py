"""Runner utilities for evaluation."""

import multiprocessing as mp
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from loguru import logger
from tqdm import tqdm

from mops_il.data_ops.video_format import uncompress_mask
from mopscasa.evaluation.environment import create_env, prepare_observation
from mopscasa.evaluation.policy_wrapper import FullPolicyWrapper
from mopscasa.observation_extender import ObservationExtender

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def run_episode(
    env_name: str,
    env_idx: int,
    policy_fn: Callable[[dict[str, Any]], np.ndarray],
    seed: int = 0,
    max_steps: int = 500,
    img_size: int = 256,
    show: bool = False,
    terminate_on_success: bool = True,
) -> tuple[bool, int]:
    """Run a single episode of evaluation.

    Args:
        env_name: Name of the environment.
        policy_fn: Function that takes an observation dictionary and returns an action.
        seed: Random seed.
        max_steps: Maximum steps per episode.
        img_size: Image size for observation.
        show: Whether to render the environment.
        terminate_on_success: Whether to terminate the episode immediately upon success.

    Returns:
        A tuple containing (success, steps).
    """
    env = create_env(env_name, seed=seed, img_size=img_size, show=show)
    # type ignore: ObservationExtender expects EnvRobocasa,
    # but we pass Kitchen (which is compatible)
    obs_extender = ObservationExtender(env)  # type: ignore

    try:
        obs = env.reset()
        done = False
        steps = 0
        success = False
        success_steps = max_steps

        task = env.get_ep_meta().get("lang", "N/A")

        with tqdm(
            total=max_steps, desc=f"Episode ({env_idx}-{env_name}, seed={seed})"
        ) as pbar:
            while steps < max_steps:
                if done and terminate_on_success:
                    break

                obs_dict = prepare_observation(obs, obs_extender, img_size=img_size)
                obs_dict["task"] = task

                action = policy_fn(obs_dict)

                if not done:
                    # type ignore: env.step returns 4 values
                    obs, _, done, _ = env.step(action)  # type: ignore
                    if show:
                        env.render()

                steps += 1
                pbar.update(1)

                # pylint: disable=protected-access
                if env._check_success():  # type: ignore
                    if not success:
                        success = True
                        success_steps = steps

                    if terminate_on_success:
                        return True, steps

        return success, success_steps

    finally:
        env.close()


def _validate_worker_observations(obs_list: list[dict]) -> None:
    for i, obs in enumerate(obs_list):
        if not isinstance(obs, dict):
            raise RuntimeError(
                f"Worker {i} failed or returned invalid observation: {obs}"
            )


def _normalize_image_tensor(tensor: torch.Tensor, key: str) -> torch.Tensor:
    # Handle image format: (B, H, W, C) -> (B, C, H, W)
    if tensor.ndim != 4:
        return tensor

    tensor = tensor.permute(0, 3, 1, 2)
    tensor = v2.ToDtype(torch.float32, scale=True)(tensor)

    # Normalize images, NOT affordance masks using ImageNet stats
    if "affordance" not in key:
        tensor = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)

    return tensor


def batch_observations(
    obs_list: list[dict], device: str, uncompress: bool = False
) -> dict:
    """Batch a list of observation dictionaries."""
    batched: dict[str, Any] = {}

    _validate_worker_observations(obs_list)

    # Assume all dicts have same keys
    first_obs = obs_list[0]

    for k, v in first_obs.items():
        if isinstance(v, (np.ndarray, list)):
            arrays = [o[k] for o in obs_list]
            stacked = np.stack(arrays, axis=0)

            tensor = torch.tensor(stacked, device=device)
            batched[k] = _normalize_image_tensor(tensor, k)
        elif isinstance(v, str):
            batched[k] = [o[k] for o in obs_list]

        if uncompress and "_segmentation_affordance" in k and "_is_pad" not in k:
            batched[k] = uncompress_mask(batched[k])

    return batched


def worker_process(
    remote: Connection,
    env_name: str,
    env_idx: int,
    seed: int,
    max_steps: int,
    img_size: int,
    show: bool,
) -> None:
    """Worker process for running a single environment."""

    def policy_fn(obs: dict[str, Any]) -> np.ndarray:
        remote.send(obs)
        return remote.recv()

    try:
        success, steps = run_episode(
            env_name=env_name,
            env_idx=env_idx,
            policy_fn=policy_fn,
            seed=seed,
            max_steps=max_steps,
            img_size=img_size,
            show=show,
            terminate_on_success=False,
        )
        remote.send((success, steps))
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Worker failed seed={seed}: {e}")
        remote.send((False, 0))
    finally:
        remote.close()


def _log_task_result(env_name: str, seed: int, success: bool, steps: int) -> None:
    if success:
        logger.info(f"Task {env_name} (seed={seed}) succeeded in {steps} steps")
    else:
        logger.info(f"Task {env_name} (seed={seed}) failed")


def _run_task(
    task: dict[str, Any],
    policy: FullPolicyWrapper,
    device: str,
    max_steps: int,
    img_size: int,
    show: bool,
    terminate_on_success: bool,
) -> tuple[bool, int]:
    policy.reset()

    def policy_fn(obs: dict[str, Any]) -> np.ndarray:
        batched_obs = batch_observations([obs], device, policy.uncompress_aff)
        action = policy.select_action(batched_obs)
        return action.cpu().numpy()[0]

    return run_episode(
        env_name=task["env_name"],
        env_idx=task["env_idx"],
        policy_fn=policy_fn,
        seed=task["seed"],
        max_steps=max_steps,
        img_size=img_size,
        show=show,
        terminate_on_success=terminate_on_success,
    )


def run_main_thread_evaluation(
    tasks: list[dict[str, Any]],
    policy: FullPolicyWrapper,
    device: str,
    max_steps: int = 500,
    show: bool = False,
    img_size: int = 256,
) -> list[tuple[str, bool]]:
    """Run evaluation sequentially in the main thread.

    Args:
        tasks: List of task dictionaries containing 'env_name', 'env_idx', 'seed'.
        policy: The policy to evaluate.
        device: Device to run the policy on.
        max_steps: Maximum steps per episode.
        show: Whether to render the environment.
        img_size: Image size for observation.

    Returns:
        List of tuples (env_name, success).
    """
    results = []

    logger.info(f"Running {len(tasks)} tasks sequentially in the main thread")

    for i, task in enumerate(tasks):
        logger.info(
            f"Starting task {i + 1}/{len(tasks)}: {task['env_name']} (seed={task['seed']})"
        )
        try:
            success, steps = _run_task(
                task,
                policy,
                device,
                max_steps,
                img_size,
                show,
                terminate_on_success=True,
            )

            results.append((task["env_name"], success))
            _log_task_result(task["env_name"], task["seed"], success, steps)

        except Exception as e:
            logger.error(
                f"Task {task['env_name']} (seed={task['seed']}) failed with error: {e}"
            )
            raise e

    return results


def run_parallel_evaluation(
    tasks: list[dict[str, Any]],
    policy: FullPolicyWrapper,
    device: str,
    num_workers: int,
    max_steps: int = 500,
    show: bool = False,
    img_size: int = 256,
) -> list[tuple[str, bool]]:
    """Run evaluation in parallel for a list of tasks.

    Args:
        tasks: List of task dictionaries containing 'env_name', 'env_idx', 'seed'.
        policy: The policy to evaluate.
        device: Device to run the policy on.
        num_workers: Number of parallel workers.
        max_steps: Maximum steps per episode.
        show: Whether to render the environment.
        img_size: Image size for observation.

    Returns:
        List of tuples (env_name, success).
    """
    if num_workers == 1:
        return run_main_thread_evaluation(
            tasks, policy, device, max_steps, show, img_size
        )

    results = []

    # Use spawn context for safety with PyTorch/CUDA
    ctx = mp.get_context("spawn")

    num_tasks = len(tasks)
    num_batches = (num_tasks + num_workers - 1) // num_workers
    logger.info(
        f"Running {num_tasks} tasks in {num_batches} batches of up to {num_workers} workers."
    )

    # Process tasks in batches of size num_workers
    for i in range(0, num_tasks, num_workers):
        current_batch_tasks = tasks[i : i + num_workers]
        current_batch_size = len(current_batch_tasks)

        logger.info(
            f"Starting batch {i // num_workers + 1}/{num_batches} with {current_batch_size} tasks"
        )

        pipes = [ctx.Pipe() for _ in range(current_batch_size)]
        parents, children = zip(*pipes, strict=True)

        processes = []
        for j, task in enumerate(current_batch_tasks):
            p = ctx.Process(
                target=worker_process,
                args=(
                    children[j],
                    task["env_name"],
                    task["env_idx"],
                    task["seed"],
                    max_steps,
                    img_size,
                    show,
                ),
            )
            p.start()
            processes.append(p)

        try:
            policy.reset()
            for _step in range(max_steps):
                # Receive observations
                obs_list = [parent.recv() for parent in parents]

                # Batch observations
                batched_obs = batch_observations(
                    obs_list, device, policy.uncompress_aff
                )

                # Policy inference
                actions = policy.select_action(batched_obs)
                actions_np = actions.cpu().numpy()

                # Send actions
                for j, parent in enumerate(parents):
                    parent.send(actions_np[j])

            # Receive results
            batch_results = [parent.recv() for parent in parents]
            for j, (success, steps) in enumerate(batch_results):
                task = current_batch_tasks[j]
                results.append((task["env_name"], success))
                _log_task_result(task["env_name"], task["seed"], success, steps)

        finally:
            for p in processes:
                p.join()

    return results
