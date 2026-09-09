"""
Sparse-reward buzz_wire scenario.

Unlike navigation/balance/transport, the sparse reward for buzz_wire cannot be
a TorchRL transform: the success predicate is the BALL's distance to the goal,
and the ball's position is not part of the agent observation (per-agent obs is
[pos, vel, pos - goal]; the midpoint-of-agents proxy errs by up to the rod
length 0.25, unusable against the native 0.01 tolerance). This subclass is passed
as a scenario *instance* to torchrl's VmasEnv (which forwards it untouched to
vmas.make_env), computing the sparse reward from simulator state.

The reward is 1 exactly on goal success and 0 otherwise.  Success also ends
the episode, so an episode return is necessarily either 0 or 1.  Native
collision termination is preserved and carries no reward.

Recorded demonstration rewards must use this same native success predicate;
it cannot be reconstructed exactly from the saved per-agent observations.
"""

from vmas.scenarios.buzz_wire import Scenario as BuzzWireScenario
from vmas.simulator.core import Agent
import torch


class SparseRewardBuzzWireScenario(BuzzWireScenario):
    def _on_goal(self) -> torch.Tensor:
        # This is the exact goal predicate used by VMAS BuzzWireScenario.done().
        return (
            torch.linalg.vector_norm(
                self.ball.state.pos - self.goal.state.pos, dim=-1
            )
            <= 0.01
        )

    def make_world(self, batch_dim, device, **kwargs):
        world = super().make_world(batch_dim, device, **kwargs)
        # torchrl builds info specs at env construction, before any reward()
        # call, so this must be a tensor from the start (never None).
        self.gt_reward = torch.zeros(batch_dim, device=device)
        return world

    def reward(self, agent: Agent):
        # Updates the cached self.pos_rew / self.collision_rew / self.rew on the
        # first agent's call; the base reward is shared across both agents.
        gt_reward = super().reward(agent)

        # Binary success reward.  Do not retain VMAS's dense position shaping:
        # remaining inside the goal for multiple simulator steps must not turn
        # one success into a return near the rollout horizon.
        final_reward = self._on_goal().to(self.pos_rew.dtype)

        self.gt_reward = gt_reward
        return final_reward

    def info(self, agent: Agent):
        info = super().info(agent)
        info["gt_reward"] = self.gt_reward
        return info

    def done(self):
        return super().done()
