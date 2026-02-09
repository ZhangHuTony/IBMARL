"""
Base class for multi-agent reinforcement learning experiments.
"""

import torch
from torch import multiprocessing
from abc import ABC, abstractmethod
import numpy as np

from src.environment.make_env import make_env

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


class BaseMARLExperiment:
    MADDPG_RESULT_KEYS = (
        "group_map_keys", 
        "episode_reward_mean_map", 
        )
    
    IBMARL_RESULTS_KEY = (
         "group_map_keys", 
        "episode_reward_mean_map", 
       "rl_action_fraction",
       "rl_only_episode_reward_mean_map",
    )

    def __init__(self, config):

        self.config = config
        #device setup
        self.device = self._setup_device()
        self._setup_seed() # Experiment Reproducibility
        
        self.env = make_env(config, self.device)
        self.render_path = config['videos_dir']

        self.experiment_type = config['exp_type']

        if self.experiment_type == "ibmarl":
            self.RESULT_KEYS = self.IBMARL_RESULTS_KEY
        elif self.experiment_type == "maddpg":
            self.RESULT_KEYS = self.MADDPG_RESULT_KEYS
        else:
            RuntimeError(f"Experiment Type: {self.experiment_type} not supported")

        self.results = self._initialize_results()




    def _setup_device(self):
        # setup device (CPU/GPU) 
        is_fork = multiprocessing.get_start_method() == 'fork'
        device =(
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
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


    def _initialize_results(self):
        results = {k: None for k in self.RESULT_KEYS}
        return results
    
    def _validate_results_complete(self):
        missing = [k for k in self.RESULT_KEYS if self.results.get(k) is None]
        if missing:
            raise RuntimeError(
                f"{self.__class__.__name__}: results incomplete: {missing}. "
            )
    
    def save_results(self):
        '''
        Takes results and saves them to needed files

        NEEDS TO BE REFACTORED BETTER THAN JUST IF STATMENTS
        '''
        self._validate_results_complete()

        data_dir = Path(self.config["data_dir"])
        metrics_path = data_dir / "metrics.csv"

        mean_map = self.results["episode_reward_mean_map"]

        if self.experiment_type == 'ibmarl':
            fraction_map = self.results.get("rl_action_fraction", {})
            rl_only_mean_map = self.results.get("rl_only_episode_reward_mean_map", {})

        # Defensive checks
        if not isinstance(mean_map, dict):
            raise TypeError(
                "episode_reward_mean_map must be a dict of "
                "{group_name: list_of_rewards}"
            )

        # Write CSV
        with open(metrics_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            # Header
            if self.experiment_type == 'ibmarl':
                writer.writerow([
                    "iteration",
                    "group",
                    "episode_reward_mean",
                    "rl_action_fraction",
                    "rl_only_episode_reward_mean",
                ])
            else: 
                writer.writerow([
                    "iteration",
                    "group",
                    "episode_reward_mean",
                ])

            # Rows
            for group, rewards in mean_map.items():
                
                if not isinstance(rewards, (list, tuple)):
                    raise TypeError(
                        f"Rewards for group '{group}' must be a list or tuple."
                    )
                
                if self.experiment_type == 'ibmarl':
                    fractions = fraction_map.get(group, [0.0] * len(rewards))
                    rl_only_list = rl_only_mean_map.get(group, [None] * len(rewards))
                    for iteration, (reward, fraction, rl_only) in enumerate(zip(rewards, fractions, rl_only_list)):
                        rl_only_val = "" if rl_only is None else float(rl_only)
                        writer.writerow([
                            iteration,
                            group,
                            float(reward),
                            float(fraction),
                            rl_only_val,
                        ])
                else: 
                   for iteration, (reward) in enumerate(rewards):
                        writer.writerow([
                            iteration,
                            group,
                            float(reward)
                        ])

        print(f"Saved metrics to: {metrics_path.resolve()}")

        # Save policy checkpoints to checkpoints subdir
        self.save_checkpoint()

        # Generate plot of episode_reward_mean_map over time
        self._save_rewards_plot(data_dir)

    def _save_rewards_plot(self, data_dir: Path):
        """
        Generate and save a plot of episode_reward_mean_map over time.
        For IBMARL experiments, also includes rl_only_episode_reward_mean_map.
        """
        plots_dir = Path(self.config.get("plots_dir", data_dir.parent / "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        mean_map = self.results["episode_reward_mean_map"]
        groups = list(mean_map.keys())

        n_plots = 2 if self.experiment_type == "ibmarl" else 1
        fig, axs = plt.subplots(len(groups), n_plots, figsize=(6 * n_plots, 4 * len(groups)), squeeze=False)

        for i, group in enumerate(groups):
            iterations = list(range(len(mean_map[group])))
            axs[i, 0].plot(iterations, mean_map[group], label=f"Episode reward mean ({group})")
            axs[i, 0].set_ylabel("Reward")
            axs[i, 0].set_title(f"{group}: Combined IL/RL" if self.experiment_type == "ibmarl" else group)
            axs[i, 0].legend()
            axs[i, 0].grid(True, alpha=0.3)

            if self.experiment_type == "ibmarl":
                rl_only_map = self.results.get("rl_only_episode_reward_mean_map", {})
                rl_only_list = rl_only_map.get(group, [])
                # Filter out None values for plotting
                rl_only_valid = [(idx, v) for idx, v in enumerate(rl_only_list) if v is not None]
                if rl_only_valid:
                    idxs, vals = zip(*rl_only_valid)
                    axs[i, 1].plot(idxs, vals, label=f"RL-only reward mean ({group})", color="orange")
                axs[i, 1].set_ylabel("Reward")
                axs[i, 1].set_title(f"{group}: RL-only")
                axs[i, 1].legend()
                axs[i, 1].grid(True, alpha=0.3)

        axs[-1, 0].set_xlabel("Training iterations")
        if self.experiment_type == "ibmarl":
            axs[-1, 1].set_xlabel("Training iterations")

        plt.tight_layout()
        plot_path = plots_dir / "episode_rewards.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved rewards plot to: {plot_path.resolve()}")

    def save_checkpoint(self):
        """
        Save policy checkpoints to the checkpoints subdir.
        Subclasses should override if they use different policy attributes (e.g. rl_policies).
        """
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
    def train(self):
        ...
    



   
        
        


    