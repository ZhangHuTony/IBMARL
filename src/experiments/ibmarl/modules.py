'''
holds custom torch module needed for IBMARL
'''

import torch
import torch.nn as nn
from torchrl.envs import ExplorationType
from torchrl.modules import AdditiveGaussianModule
from tensordict.nn import TensorDictSequential


def build_il_noise_modules(cfg, env, device):
    """
    Build AdditiveGaussianModule per group for IL policy exploration.
    Uses same sigma/annealing as RL exploration noise (exploration_noise config).
    """
    noise_config = cfg.get("exploration_noise", {})
    sigma_init = noise_config.get("sigma_init", 0.1)
    sigma_end = noise_config.get("sigma_end", 0.1)

    il_noise_modules = {}
    for group, _ in env.group_map.items():
        spec = env.full_action_spec_unbatched[group, "action"].to(device)
        module = AdditiveGaussianModule(
            spec=spec,
            annealing_num_steps=cfg.get("total_frames") // 2,
            action_key=(group, "action"),
            sigma_init=sigma_init,
            sigma_end=sigma_end,
        )
        il_noise_modules[group] = module
    return il_noise_modules


class OverWriteActionWithBestComb(torch.nn.Module):
    def __init__(self, parent, group:str):
        super().__init__()
        self.parent = parent
        self.group = group


    @torch.no_grad()
    def forward(self, td):
        g = self.group
        obs = td[(g, "observation")] 
        a_rl = td[(g, "action")]


        a_exec, arbiter_choice_mask, metrics = self.parent.action_arbiter.actor_proposal(g, obs, a_rl)

        td[(g, "action")] = a_exec
        td[(g, "arbiter_choice")] = arbiter_choice_mask
        # Store metrics in the batch for later logging
        # Create tensors with matching batch dimensions by matching the shape of arbiter_choice_mask
        metric_shape = arbiter_choice_mask.shape
        
        # Create tensors with the correct shape using torch.zeros and fill_
        mean_action_diff_tensor = torch.zeros(metric_shape, device=td.device, dtype=torch.float32)
        mean_action_diff_tensor.fill_(metrics['mean_action_diff'])
        mean_q_diff_tensor = torch.zeros(metric_shape, device=td.device, dtype=torch.float32)
        mean_q_diff_tensor.fill_(metrics['mean_q_diff'])
        var_q_diff_tensor = torch.zeros(metric_shape, device=td.device, dtype=torch.float32)
        var_q_diff_tensor.fill_(metrics['var_q_diff'])
        
        td[(g, "mean_action_diff")] = mean_action_diff_tensor
        td[(g, "mean_q_diff")] = mean_q_diff_tensor
        td[(g, "var_q_diff")] = var_q_diff_tensor
        return td


def build_exploration_policy_with_noise(policy, spec, cfg, group):
    """
    Build an exploration policy that wraps a clean policy with AdditiveGaussianModule.
    The noise module respects ExplorationType (no noise in MODE, noise in RANDOM).
    """
    # Get noise parameters from config, with defaults
    noise_config = cfg.get('exploration_noise', {})
    sigma_init = noise_config.get('sigma_init', 0.1)
    sigma_end = noise_config.get('sigma_end', 0.1)
    
    noise_module = AdditiveGaussianModule(
        spec=spec,
        annealing_num_steps=cfg.get('total_frames') // 2,
        action_key=(group, "action"),
        sigma_init=sigma_init,
        sigma_end=sigma_end,
    )
    
    # AdditiveGaussianModule should respect ExplorationType automatically via set_exploration_type()
    # When ExplorationType.MODE is set, it won't add noise
    # When ExplorationType.RANDOM is set, it will add noise
    return TensorDictSequential(policy, noise_module)

