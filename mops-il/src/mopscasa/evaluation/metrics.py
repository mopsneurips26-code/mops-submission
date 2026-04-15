"""Metrics tracking for evaluation."""

from typing import Any

from loguru import logger

import wandb


class EvaluationMetrics:
    """Tracks and logs evaluation metrics."""

    def __init__(self, use_wandb: bool = False) -> None:
        self.use_wandb = use_wandb
        self.results: dict[str, dict[str, Any]] = {}
        self.total_success = 0
        self.total_trials = 0

    def record_trial(self, env_name: str, success: bool) -> None:
        """Record result for a single trial."""
        if env_name not in self.results:
            self.results[env_name] = {
                "success_count": 0,
                "total_trials": 0,
                "success_rate": 0.0,
            }

        self.results[env_name]["total_trials"] += 1
        if success:
            self.results[env_name]["success_count"] += 1

        self.results[env_name]["success_rate"] = (
            self.results[env_name]["success_count"]
            / self.results[env_name]["total_trials"]
        )

        self.total_trials += 1
        if success:
            self.total_success += 1

        self._log_env(
            env_name,
            self.results[env_name]["success_count"],
            self.results[env_name]["success_rate"],
        )
        self._log_running_total()

    def record_env(self, env_name: str, success_count: int, num_trials: int) -> None:
        """Record results for a single environment."""
        success_rate = success_count / num_trials
        self.results[env_name] = {
            "success_count": success_count,
            "total_trials": num_trials,
            "success_rate": success_rate,
        }
        self.total_success += success_count
        self.total_trials += num_trials

        self._log_env(env_name, success_count, success_rate)
        self._log_running_total()

    def _log_env(self, env_name: str, success_count: int, success_rate: float) -> None:
        logger.info(f"Success Rate for {env_name}: {success_rate * 100:.2f}%")
        if self.use_wandb and wandb.run:
            wandb.log(
                {
                    f"eval/{env_name}_success_rate": success_rate,
                    f"eval/{env_name}_success_count": success_count,
                }
            )

    def _log_running_total(self) -> None:
        running_rate = (
            self.total_success / self.total_trials if self.total_trials > 0 else 0.0
        )
        if self.use_wandb and wandb.run:
            wandb.log(
                {
                    "eval/running_success_rate": running_rate,
                    "eval/running_success_count": self.total_success,
                    "eval/running_total_trials": self.total_trials,
                }
            )

    def log_summary(self) -> None:
        """Log the final summary of results."""
        logger.info("Evaluation Summary:")
        for env_name, data in self.results.items():
            logger.info(f"{env_name}: {data['success_count']} successful trials")
            logger.info(f"{env_name}: {data['success_rate'] * 100:.2f}% success rate")

        overall_rate = (
            self.total_success / self.total_trials if self.total_trials > 0 else 0.0
        )
        logger.info("Evaluation Summary:")
        logger.info(
            f"Total successful trials: {self.total_success}/{self.total_trials}"
        )
        logger.info(f"Overall success rate: {overall_rate * 100:.2f}%")

        if self.use_wandb and wandb.run:
            wandb.log(
                {
                    "eval/total_success_rate": overall_rate,
                    "eval/total_success_count": self.total_success,
                    "eval/total_trials": self.total_trials,
                }
            )
