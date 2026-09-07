"""
Sparse-reward, fixed-horizon buzz_wire scenario.

Unlike navigation/balance/transport, the sparse reward for buzz_wire cannot be
a TorchRL transform: the success predicate is the BALL's distance to the goal,
and the ball's position is not part of the agent observation (per-agent obs is
[pos, vel, pos - goal]; the midpoint-of-agents proxy errs by up to the rod
length 0.25, unusable against a 0.1 radius). Instead this subclass is passed
as a scenario *instance* to torchrl's VmasEnv (which forwards it untouched to
vmas.make_env), computing the sparse reward from simulator state.

done() is suppressed so vmas's `scenario.done() + (steps >= max_steps)` reduces
to pure horizon truncation - fixed-length episodes like sparse navigation, and
no pathological "crash early to stop collecting -1" incentive.

KEEP IN LOCKSTEP with R2BC/src/scenarios/buzz_wire/wrappers/sparse_reward.py -
the recorded demonstration rewards must match this formula exactly (same
success_threshold), and the threshold cannot be recomputed from saved
observations after the fact.
"""

from vmas.scenarios.buzz_wire import Scenario as BuzzWireScenario
from vmas.simulator.core import Agent
import torch


class SparseRewardBuzzWireScenario(BuzzWireScenario):
    def __init__(self, success_threshold: float = 0.1, out_penalty: float = -1.0):
        super().__init__()
        self.success_threshold = success_threshold
        self.out_penalty = out_penalty

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

        dist = torch.linalg.vector_norm(
            self.ball.state.pos - self.goal.state.pos, dim=-1
        )
        inside_mask = (dist < self.success_threshold).to(self.pos_rew.dtype)
        outside_mask = 1.0 - inside_mask

        # pos_rew only inside the radius (no collision term); flat penalty outside.
        final_reward = (self.pos_rew * inside_mask) + (self.out_penalty * outside_mask)

        self.gt_reward = gt_reward
        return final_reward

    def info(self, agent: Agent):
        info = super().info(agent)
        info["gt_reward"] = self.gt_reward
        return info

    def done(self):
        done = super().done()
        done.fill_(False)
        return done
