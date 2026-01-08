"""
Base class for multi-agent reinforcement learning experiments.
"""

import torch
from torch import multiprocessing


from src.base_experiment import BaseExperiment



class BaseMARLExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)

        #device setup
        self.device = self._setup_device()


    def _setup_device(self):
        # setup device (CPU/GPU) 
        is_fork = multiprocessing.get_start_method() == 'fork'
        device =(
            torch.device(0)
            if torch.cuda.is_available() and not is_fork
            else torch.device("cpu")
        )

        return device
    
    def _setup_environment(self):
        num_vmas_env = ()