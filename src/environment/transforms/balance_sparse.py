import torch
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms import Transform


class BalanceSparseReward(Transform):
    """Emit a binary reward and end an episode when the ball reaches the goal."""

    def __init__(
        self,
        *,
        group: str = "agents",
        success_threshold,
    ):
        # No in_keys/out_keys: this transform updates reward and termination
        # together from the complete TensorDict produced by the environment.
        super().__init__()

        self.group = group
        self.success_threshold = success_threshold

    def _compute_dist(self, obs: torch.Tensor) -> torch.Tensor:
        # Balance observations contain package_pos - goal_pos at indices 8:10.
        # This value is identical for every agent, so use the first agent to
        # produce one success predicate per parallel environment.
        package_to_goal = obs[..., 0, 8:10]
        return torch.linalg.vector_norm(package_to_goal, dim=-1)

    def _call(self, td):
        """Replace the dense reward and terminate successful environments."""
        obs = td.get((self.group, "observation"))
        current_reward = td.get((self.group, "reward"))

        success = self._compute_dist(obs) < self.success_threshold

        # All agents share the balance objective and therefore receive the
        # same 0/1 reward.
        reward_success = success
        while reward_success.ndim < current_reward.ndim:
            reward_success = reward_success.unsqueeze(-1)
        reward = reward_success.to(current_reward.dtype).expand_as(current_reward)
        td.set((self.group, "reward"), reward)

        # Preserve native terminal conditions (for example, dropping the ball)
        # while also ending an environment immediately on a rewarded step.
        for key in ("done", "terminated"):
            terminal = td.get(key, None)
            if terminal is None:
                continue

            terminal_success = success
            while terminal_success.ndim < terminal.ndim:
                terminal_success = terminal_success.unsqueeze(-1)
            td.set(key, terminal | terminal_success.expand_as(terminal))

        return td

    def transform_reward_spec(self, reward_spec):
        """Keep the transformed reward compatible with the environment spec."""
        if isinstance(reward_spec, CompositeSpec):
            current_spec = reward_spec[self.group, "reward"]
            reward_spec[self.group, "reward"] = UnboundedContinuousTensorSpec(
                shape=current_spec.shape,
                device=current_spec.device,
                dtype=current_spec.dtype,
            )
        return reward_spec
