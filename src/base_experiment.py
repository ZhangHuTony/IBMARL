"""
Base Experiment Class for RL Experiments.
"""

import torch

class BaseExperiment:
    def __init__(self, config: dict):
        self.config = config
        self.results = self._init_results()
        self._setup_seed()



    def _setup_seed(self):
        self.seed = self.config.get('seed', None) 

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            print(f"Random seed set to: {self.seed}")


    
    def _setup_environment(self):
        # setup VMAS environment here
        pass

    def _setup_policy(self):
        # setup policy network here
        pass

    def _setup_critic(self):
        # setup critic network here
        pass

    def _init_results(self):
        self.results = {
            "training_rewards": [],
            "evaluation_rewards": []
        }

            

    def train(self) -> str:
        raise NotImplementedError("Subclasses should implement this method.")
        return "Training Done"
    
    def evaluate(self):
        raise NotImplementedError("Subclasses should implement this method.")