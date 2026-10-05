"""
Balance's gt-in-radius sparse reward (the legacy -1/step family, see CONTEXT.md
"Sparse Navigation"): the native VMAS reward -- 100 x the per-step decrease in
package-goal distance plus the -10 floor-contact penalty -- while the package is
within ``gt_radius`` of the goal, a flat ``out_penalty`` (-1) per step otherwise.
With the episode end suppressed (scenarios/balance_fixed_horizon.py) a return
reads as -(steps spent outside the basin) plus the shaping earned inside it.

The predicate is the package-goal offset at observation indices 8:10
(``package.pos - goal.pos`` in vmas 1.5.2's balance observation), identical for
every agent, so one distance per sub-env decides the whole team's reward.  A
TorchRL ``Transform._call`` runs on the post-step tensordict, so that offset is
the NEXT observation's and the reward paid for transition t is conditioned on
the state the step led to -- the same convention analysis/relabel_demos_sparse.py
applies to the recorded demonstrations, so online and demo rewards agree.

NOT a binary terminal schema: balance has no ``binary_terminal_reward`` branch
(Connor's ``binary_balance_exp`` branch is a different experiment).
"""

import torch
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms import Transform


class BalanceSparseReward(Transform):
    def __init__(
        self,
        *,
        group: str = "agents",
        success_threshold: float,
        out_penalty: float = -1.0,
        rel_goal_slice: slice = slice(8, 10),
    ):
        # No in_keys/out_keys: the base class's _apply_transform path is bypassed
        # and _call rewrites the reward from the whole tensordict instead.
        super().__init__()
        self.group = group
        self.success_threshold = float(success_threshold)
        self.out_penalty = float(out_penalty)
        self.rel_goal_slice = rel_goal_slice

    def _compute_dist(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Package-goal distance per sub-env, shape ``obs.shape[:-2]``.

        ``obs`` is ``[..., n_agents, obs_dim]``; the offset is read from agent 0
        because every agent observes the same package and goal.
        """
        rel = obs[..., 0, self.rel_goal_slice]
        return torch.linalg.vector_norm(rel, dim=-1)

    def _call(self, td):
        obs = td.get((self.group, "observation"))
        gt_reward = td.get((self.group, "reward"))  # [..., n_agents, 1]

        dist = self._compute_dist(obs)  # [...]
        # Broadcast the per-env predicate over the agent and reward axes
        # explicitly.  The earlier ``unsqueeze(-1)`` produced [n_envs, 1], which
        # right-aligns against [n_envs, n_agents, 1] by pairing n_envs with
        # n_agents: a RuntimeError at 10 sub-envs, and silently cross-wired
        # (agent a read env a's predicate) at the 3 sub-envs it was written for.
        inside = (dist < self.success_threshold)[..., None, None].to(gt_reward.dtype)
        new_reward = gt_reward * inside + self.out_penalty * (1.0 - inside)

        td.set((self.group, "reward"), new_reward)
        return td

    def transform_reward_spec(self, reward_spec):
        if isinstance(reward_spec, CompositeSpec):
            curr_spec = reward_spec[self.group, "reward"]
            reward_spec[self.group, "reward"] = UnboundedContinuousTensorSpec(
                shape=curr_spec.shape,
                device=curr_spec.device,
                dtype=curr_spec.dtype,
            )
        return reward_spec
