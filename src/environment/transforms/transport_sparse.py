import torch
from torchrl.envs.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

class TransportSparseReward(Transform):
    """
    Sparse reward for transport:
    reward = 1 if every package is on its goal, else 0.

    Success is read off the ``on_goal`` flag VMAS already writes into the
    observation rather than a distance threshold, for two reasons:

    * It is the same predicate as ``transport.Scenario.done()``
      (``all(package.on_goal)``), so the success reward and the episode
      termination stay in sync.  A ``gt_radius``-style threshold would fire
      before or after the episode actually ends.
    The success transition also terminates the episode, keeping every episode
    return in the interval [0, 1].

    Like the navigation/balance transforms, this overrides _call directly to
    avoid the NotImplementedError caused by the base Transform class trying to
    use _apply_transform.
    """

    def __init__(
        self,
        *,
        group: str = "agents",
        n_packages: int = 1,
    ):
        # We initialize the base class without in_keys/out_keys.
        # This prevents the base class from trying to run its automated
        # _apply_transform logic, which is what causes the error.
        super().__init__()

        self.group = group
        self.n_packages = n_packages

        # Per-agent observation layout is
        #   pos(2) + vel(2) + n_packages * [pkg-goal(2), pkg-agent(2), pkg_vel(2), on_goal(1)]
        # so package i's on_goal flag sits at 4 + 7*i + 6.  With the default
        # single package that is index 10 of an 11-dim observation.
        self.on_goal_indices = [4 + 7 * i + 6 for i in range(n_packages)]

    def _compute_success(self, obs: torch.Tensor) -> torch.Tensor:
        # (..., n_agents, n_packages) -> (..., n_agents)
        on_goal = obs[..., self.on_goal_indices]
        return (on_goal > 0.5).all(dim=-1).to(obs.dtype)

    def _call(self, td):
        """
        Manually calculate the reward and update the TensorDict.
        This is called during the environment step.
        """
        # The observation we need is located in the same TensorDict
        # that the environment just populated during its internal _step.
        obs = td.get((self.group, "observation"))

        current_reward = td.get((self.group, "reward"))

        # Reshape to match reward specs (..., n_agents, 1)
        success_by_agent = self._compute_success(obs)
        success = success_by_agent.all(dim=-1)
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
