import torch
from torchrl.envs.transforms import Transform


class NavigationSparseReward(Transform):
    """
    Sparse reward for navigation:
    reward = 1 if agent is on goal, else 0.

    Assumes the observation contains either:
    - distance to goal at a known index, OR
    - relative goal vector (dx, dy) at a known slice.
    """

    def __init__(
        self,
        *,
        group: str = "agents",
        success_threshold: float = 0.05,
        distance_index: int | None = None,
        rel_goal_slice: slice | None = None,
    ):
        super().__init__(in_keys=[(group, "observation"), ("next", group, "reward")])
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
        obs = td.get((self.group, "observation"))

        dist = self._compute_dist(obs)
        success = (dist < self.success_threshold).to(obs.dtype)

        # reward shape: (..., n_agents, 1)
        reward = success.unsqueeze(-1)

        td.set(("next", self.group, "reward"), reward)
        return td
