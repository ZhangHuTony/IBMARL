"""Binary sparse-reward balance scenario using VMAS's native goal test."""

import torch
from vmas.scenarios.balance import Scenario as BalanceScenario
from vmas.simulator.core import Agent


class SparseRewardBalanceScenario(BalanceScenario):
    """Reward package/goal overlap once and terminate on that transition."""

    def _on_goal(self) -> torch.Tensor:
        # This is the exact predicate used by VMAS BalanceScenario.done().
        return self.world.is_overlapping(self.package, self.package.goal)

    def reward(self, agent: Agent):
        # Preserve the base scenario's state updates, especially
        # ``on_the_ground``, which is also a native terminal condition.
        dense_reward = super().reward(agent)
        return self._on_goal().to(dense_reward.dtype)

    def done(self):
        # The base implementation is on_the_ground OR package/goal overlap.
        return super().done()
