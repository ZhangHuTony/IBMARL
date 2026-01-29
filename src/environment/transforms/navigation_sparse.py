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

        #----1.0 if close otherwise 0-------------------#
        
        # Calculate sparse reward: 1.0 if close enough, else 0.0
        # success = (dist < self.success_threshold).to(obs.dtype)
        
        # # Reshape to (..., n_agents, 1) to match TorchRL reward specs
        # reward = success.unsqueeze(-1)

        # # Overwrite the default reward with our sparse version
        # td.set((self.group, "reward"), reward)
        #--------------------------------------------------------#

        # ------------------ gt if close otherwise -1 ------------#
        out_penalty = -0.1

        gt_reward = td.get((self.group, "reward"))

        # Calculate success mask: 1.0 if close enough, else 0.0
        success_mask = (dist < self.success_threshold).to(obs.dtype)
        
        # Reshape to match reward specs (..., n_agents, 1)
        success_mask = success_mask.unsqueeze(-1)

        # Calculate failure mask: 0.0 if close enough, else 1.0
        failure_mask = 1.0 - success_mask

        # Apply logic:
        # 1. Keep gt_reward where success_mask is 1
        # 2. Add -1.0 where failure_mask is 1 (which acts as the "else" condition)
        new_reward = (gt_reward * success_mask) + (out_penalty * failure_mask)

        # Overwrite the default reward
        td.set((self.group, "reward"), new_reward)

        #----------------------------------------------------------_#
        
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