import torch
from torchrl.envs.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

class NavigationSparseReward(Transform):
    """
    Sparse reward for navigation:
    reward = 1 if agent is on goal, else 0.

    This version overrides _call directly to avoid the NotImplementedError
    caused by the base Transform class trying to use _apply_transform.
    """

    def __init__(
        self,
        *,
        group: str = "agents",
        success_threshold,
        distance_index: int | None = None,
        rel_goal_slice: slice | None = None,
    ):
        # We initialize the base class without in_keys/out_keys.
        # This prevents the base class from trying to run its automated 
        # _apply_transform logic, which is what causes the error.
        super().__init__()
        
        self.group = group
        self.success_threshold = success_threshold
        self.distance_index = distance_index
        self.rel_goal_slice = rel_goal_slice

        if distance_index is None and rel_goal_slice is None:
            raise ValueError(
                "NavigationSparseReward requires distance_index or rel_goal_slice"
            )

    def _compute_dist(self, obs: torch.Tensor) -> torch.Tensor:
        if self.distance_index is not None:
            return obs[..., self.distance_index].abs()

        rel = obs[..., self.rel_goal_slice]  # (..., n_agents, 2)
        return torch.linalg.vector_norm(rel, dim=-1)

    def _call(self, td):
        """
        Manually calculate the reward and update the TensorDict.
        This is called during the environment step.
        """
        # The observation we need is located in the same TensorDict 
        # that the environment just populated during its internal _step.
        obs = td.get((self.group, "observation"))

        dist = self._compute_dist(obs)

        current_reward = td.get((self.group, "reward"))

        # Navigation is a cooperative task: the episode succeeds once every
        # agent is on its goal.  Emit one shared success reward on that terminal
        # transition, rather than repeatedly rewarding agents that arrived
        # before their teammates.
        success = (dist < self.success_threshold).all(dim=-1)
        reward_success = success
        while reward_success.ndim < current_reward.ndim:
            reward_success = reward_success.unsqueeze(-1)
        reward = reward_success.to(current_reward.dtype).expand_as(current_reward)
        td.set((self.group, "reward"), reward)

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
        """
        Ensures check_env_specs (and the environment in general) knows 
        that we are providing a valid reward specification.
        """
        if isinstance(reward_spec, CompositeSpec):
            # We ensure the reward spec for our group matches our output
            curr_spec = reward_spec[self.group, "reward"]
            reward_spec[self.group, "reward"] = UnboundedContinuousTensorSpec(
                shape=curr_spec.shape,
                device=curr_spec.device,
                dtype=curr_spec.dtype
            )
        return reward_spec
