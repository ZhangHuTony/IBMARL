"""
Base class for multi-agent reinforcement learning experiments.
"""

import torch
from torch import multiprocessing


from src.environment.make_env import make_env


class BaseMARLExperiment:
    def __init__(self, config):

        self.config = config
        #device setup
        self.device = self._setup_device()

        self._setup_device
        
        self.env = make_env(config, self.device)


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


    