import torch
from torchrl.envs.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

class BalanceSparseReward(Transform):
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
    ):
        # We initialize the base class without in_keys/out_keys.
        # This prevents the base class from trying to run its automated 
        # _apply_transform logic, which is what causes the error.
        super().__init__()
        
        self.group = group
        self.success_threshold = success_threshold

    def _compute_dist(self, obs: torch.Tensor) -> torch.Tensor:
        # print("OBS SHAPE", obs.shape)
        rel = obs[..., 8:10]  # (..., n_agents, 2)
        rel = rel[:, 0]
        # print("REL", rel)
        # print("REL SHAPE", rel.shape)
        dist = torch.linalg.vector_norm(rel, dim=-1)
        # print("DIST", dist)
        # print("REL NORM", torch.linalg.vector_norm(rel, dim=-1))
        return dist

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
        out_penalty = -1

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

        # if torch.rand(1) < 0.01: 
        #     # Select First Batch, First Agent (index [0, 0])
        #     d_val = dist[0, 0].item() 
        #     r_val = new_reward[0, 0].item()
        #     gt_val = gt_reward[0, 0].item()
            
        #     print(f"--- Debug Reward (Agent 0, Env 0) ---")
        #     print(f"Dist: {d_val:.4f} | Threshold: {self.success_threshold}")
        #     print(f"GT Reward: {gt_val:.4f} | Out Penalty: {out_penalty}")
        #     print(f"Final Reward: {r_val:.4f}")
            
        #     # Sanity Check Alert
        #     if d_val > self.success_threshold and r_val > -self.success_threshold:
        #          print("WARNING: Penalty is not harsh enough! Agent might stay outside.")

        # Overwrite the default reward

        td.set((self.group, "reward"), new_reward)

        # print("NEW_REWARD", new_reward)
        # print("GT_REWARD", gt_reward)
        # print("Difference", new_reward - gt_reward)
        # print("Success_mask", success_mask)
        # print("Failure_mask", failure_mask)

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