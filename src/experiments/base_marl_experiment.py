"""
Base class for multi-agent reinforcement learning experiments.
"""

import torch
from abc import abstractmethod
import numpy as np

from tensordict.nn import TensorDictSequential
from torchrl.envs import ExplorationType, set_exploration_type

from src.environment.make_env import make_env
from src.util.metrics_logger import MetricsLogger

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BaseMARLExperiment:

    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._setup_seed()

        self.env = make_env(config, self.device)
        self.render_path = config['videos_dir']
        self.experiment_type = config['exp_type']

        data_dir = Path(config["data_dir"])
        self.metrics_logger = MetricsLogger(data_dir / "metrics.csv")

    def _setup_device(self):
        device = (
            torch.device(0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return device

    def _setup_seed(self):
        self.seed = self.config.get('seed', None)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        print(f"Setting seed to: {self.config.get('seed', None)}")

    def evaluate(self, n_episodes: int = 20) -> dict:
        """
        Deterministic evaluation: run the learned policy (no exploration noise)
        for *n_episodes* complete episodes and return mean episode reward per group.

        Uses self.rl_policies (IBMARL) or self.policies (MADDPG / RLFD / RFT).
        """
        policies = getattr(self, "rl_policies", None) or getattr(self, "policies", None)
        if policies is None:
            raise RuntimeError("No policies found (expected self.policies or self.rl_policies)")

        eval_policy = TensorDictSequential(*policies.values())
        eval_policy.eval()

        horizon = self.config.get("horizon", 100)
        max_steps = horizon * (n_episodes + 5)

        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                out = self.env.rollout(max_steps, policy=eval_policy)

        mean_reward_by_group = {}
        for group in self.env.group_map.keys():
            done = out.get(("next", group, "done"))
            episode_rewards = out.get(("next", group, "episode_reward"))[done]
            if episode_rewards.numel() == 0:
                mean_reward_by_group[group] = 0.0
                continue
            n = min(n_episodes, episode_rewards.shape[0])
            mean_reward_by_group[group] = episode_rewards[:n].float().mean().item()

        return mean_reward_by_group

    def save_results(self):
        self.metrics_logger.save()
        print(f"Saved metrics to: {self.metrics_logger.path.resolve()}")
        self.save_checkpoint()
        self._save_rewards_plot()

    def _save_rewards_plot(self):
        data_dir = Path(self.config["data_dir"])
        plots_dir = Path(self.config.get("plots_dir", data_dir.parent / "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        groups = list(self.env.group_map.keys())
        has_eval = "eval_reward_mean" in self.metrics_logger.columns

        n_plots = 2 if has_eval else 1
        fig, axs = plt.subplots(
            len(groups), n_plots,
            figsize=(6 * n_plots, 4 * len(groups)),
            squeeze=False,
        )

        for i, group in enumerate(groups):
            iterations = self.metrics_logger.get_values("iteration", group=group)
            rewards = self.metrics_logger.get_values("episode_reward_mean", group=group)
            axs[i, 0].plot(iterations, rewards, label=f"In-batch reward ({group})")
            axs[i, 0].set_ylabel("Reward")
            axs[i, 0].set_title(f"{group}: Training batch")
            axs[i, 0].legend()
            axs[i, 0].grid(True, alpha=0.3)

            if has_eval:
                eval_vals = self.metrics_logger.get_values(
                    "eval_reward_mean", group=group
                )
                valid = [
                    (it, v)
                    for it, v in zip(iterations, eval_vals)
                    if v is not None
                ]
                if valid:
                    idxs, vals = zip(*valid)
                    axs[i, 1].plot(
                        idxs, vals,
                        label=f"Eval reward ({group})",
                        color="orange",
                    )
                axs[i, 1].set_ylabel("Reward")
                axs[i, 1].set_title(f"{group}: Eval (20 episodes)")
                axs[i, 1].legend()
                axs[i, 1].grid(True, alpha=0.3)

        axs[-1, 0].set_xlabel("Training iterations")
        if has_eval:
            axs[-1, 1].set_xlabel("Training iterations")

        plt.tight_layout()
        plot_path = plots_dir / "episode_rewards.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved rewards plot to: {plot_path.resolve()}")

    def save_checkpoint(self):
        policies = getattr(self, "rl_policies", None) or getattr(self, "policies", None)
        if policies is None:
            raise RuntimeError("No policies to save (expected self.policies or self.rl_policies)")

        check_dir = Path(self.config["check_dir"])
        check_dir.mkdir(parents=True, exist_ok=True)

        state_dicts = {group: policy.state_dict() for group, policy in policies.items()}
        checkpoint_path = check_dir / "policy_checkpoint.pt"
        torch.save(state_dicts, checkpoint_path)
        print(f"Saved policy checkpoint to: {checkpoint_path.resolve()}")

    def render_policy(self):
        raise NotImplementedError

    @abstractmethod
    def train(self) -> str:
        ...
