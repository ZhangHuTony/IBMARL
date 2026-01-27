"""
Base class for multi-agent reinforcement learning experiments.
"""

import torch
from torch import multiprocessing
from abc import ABC, abstractmethod


from src.environment.make_env import make_env

from pathlib import Path
import csv


class BaseMARLExperiment:
    RESULT_KEYS = (
        "group_map_keys", 
        "episode_reward_mean_map", 
       "rl_action_fraction"
        )

    def __init__(self, config):

        self.config = config
        #device setup
        self.device = self._setup_device()

        
        self.env = make_env(config, self.device)

        self.results = self._initialize_results()

        self.render_path = config['videos_dir']




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

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            print(f"Random seed set to: {self.seed}")

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
        '''
        self._validate_results_complete()

        data_dir = Path(self.config["data_dir"])
        metrics_path = data_dir / "metrics.csv"

        mean_map = self.results["episode_reward_mean_map"]
        fraction_map = self.results.get("rl_action_fraction", {})

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
            writer.writerow([
                "iteration",
                "group",
                "episode_reward_mean",
                "rl_action_fraction"
            ])

            # Rows
            for group, rewards in mean_map.items():
                fractions = fraction_map.get(group, [0.0] * len(rewards))
                if not isinstance(rewards, (list, tuple)):
                    raise TypeError(
                        f"Rewards for group '{group}' must be a list or tuple."
                    )

                for iteration, (reward,fraction) in enumerate(zip(rewards,fractions)):
                #for iteration, (reward) in enumerate(rewards):

                    writer.writerow([
                        iteration,
                        group,
                        float(reward),
                       float(fraction)
                    ])

        print(f"Saved metrics to: {metrics_path.resolve()}")

    def save_checkpoint(self):
        '''
        Generic checkpoint saver.
        Requires: self.policies is a dict {group: torch.nn.Module or TensorDictModule}
        Optionally saves self.critics and self.optimisers if present.
        '''

        check_dir = self.config["check_dir"]

    def render_policy(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        ...
    



   
        
        


    