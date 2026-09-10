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
            annealing_num_steps=cfg.get(
                "exploration_annealing_frames", cfg.get("total_frames") // 2
            ),
            action_key=(group, "action"),
            sigma_init=sigma_init,
            sigma_end=sigma_end,
        )
        il_noise_modules[group] = module
    return il_noise_modules


class OverWriteActionWithBestComb(torch.nn.Module):
    """
    Replace the RL action in *td* with the arbiter's choice between the RL and
    teacher proposals.

    Takes the :class:`ActionArbiter` directly rather than the owning experiment:
    that is all it ever used, and it lets the evaluation harness in
    ``analysis/eval_checkpoints.py`` -- which rebuilds an arbiter without an
    experiment around it -- reuse this module.

    With *greedy* set this is the evaluation policy: no exploration noise on the
    teacher proposal, a deterministic ensemble reduction and argmax selection.
    The arbiter returns no metrics in that mode, so only the action and the
    choice mask are written back; the mask survives ``env.rollout`` and is what
    the eval RL-action fraction is computed from.
    """

    def __init__(self, arbiter, group: str, greedy: bool = False):
        super().__init__()
        self.arbiter = arbiter
        self.group = group
        self.greedy = greedy

    @torch.no_grad()
    def forward(self, td):
        g = self.group
        obs = td[(g, "observation")]
        a_rl = td[(g, "action")]

        a_exec, arbiter_choice_mask, metrics = self.arbiter.actor_proposal(
            g, obs, a_rl, greedy=self.greedy
        )

        td[(g, "action")] = a_exec
        td[(g, "arbiter_choice")] = arbiter_choice_mask

        if metrics is None:
            return td

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


def build_eval_policies(arbiter, rl_policies, env):
    """
    Per-group evaluation policy: clean RL actor -> greedy arbiter.

    The RL+IL protocol from IBRL -- the critic picks greedily between the RL
    actor's action and the teacher's action.  Deliberately *not* the collector's
    policy: no exploration-noise module, and the arbiter runs in greedy mode.

    Shares parameters with *rl_policies* rather than copying them, so it tracks
    training automatically and must not be added to the resume state.
    """
    return {
        group: TensorDictSequential(
            rl_policies[group],
            OverWriteActionWithBestComb(arbiter, group, greedy=True),
        )
        for group in env.group_map.keys()
    }


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
        annealing_num_steps=cfg.get(
            "exploration_annealing_frames", cfg.get("total_frames") // 2
        ),
        action_key=(group, "action"),
        sigma_init=sigma_init,
        sigma_end=sigma_end,
    )
    # AdditiveGaussianModule registers its sigma/mean/std buffers on the CPU.
    # SyncDataCollector casts the policy to its device by DEEP-COPYING it when
    # any parameter or buffer lives elsewhere, and that copy took the
    # ActionArbiter (a plain attribute, outside torchrl's weight mapping) with
    # it: critics frozen at initialisation for the acting arbiter, and noise
    # annealing that never reached the collector.  With every buffer on the
    # policy's device torchrl hands the collector this very object;
    # build_data_collector verifies that it did.
    noise_module = noise_module.to(next(policy.parameters()).device)

    # AdditiveGaussianModule should respect ExplorationType automatically via set_exploration_type()
    # When ExplorationType.MODE is set, it won't add noise
    # When ExplorationType.RANDOM is set, it will add noise
    return TensorDictSequential(policy, noise_module)
