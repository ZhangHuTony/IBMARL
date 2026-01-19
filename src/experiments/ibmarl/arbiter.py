'''
class used to find the best action combination given an IL policy and Critic
'''
import torch

class ActionArbiter:
    def __init__(self, il_policy, critics, env, device):
        pass

    def il_action(self, group, obs):
        pass

    def best_act_comb(self, group, obs, a_rl) -> torch.Tensor:
        pass

    def best_next_act_comb(self, group, next_obs, next_a_rl=None):
        pass