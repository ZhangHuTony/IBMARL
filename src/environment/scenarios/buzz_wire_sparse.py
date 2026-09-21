"""
Sparse buzz_wire scenario: reward and termination live at scenario level.

Unlike navigation/balance/transport, the sparse reward for buzz_wire cannot be
a TorchRL transform: the success predicate is the BALL's distance to the goal,
and the ball's position is not part of the agent observation (per-agent obs is
[pos, vel, pos - goal]; the midpoint-of-agents proxy errs by up to the rod
length 0.25, unusable against the native 0.01 tolerance). This subclass is passed
as a scenario *instance* to torchrl's VmasEnv (which forwards it untouched to
vmas.make_env), computing the reward from simulator state.

Two schemas, selected by ``binary_terminal`` (config ``binary_terminal_reward``):

* binary_terminal=True -- the task definition: reward 1.0 to both agents on the
  step the ball first comes within ``success_threshold`` of the goal, 0.0 on
  every other step, and ``done()`` fires on that step.  Wire/floor contact ALSO
  terminates, with reward 0 -- the buzz-wire rule, and not optional: without it
  a constant upward push drifts the ball along the walls into the basin in ~40%
  of episodes (measured 24/64, every one after a collision; the teacher collides
  before success in 5/59), so an untrained actor scored ~0.95 and "ever reached
  the basin" stopped measuring skill.  The legacy schema hid the same drift as a
  very late arrival (-177 at initialisation).  A step on which the ball enters
  the basin while touching a wall counts as a failure.  Time limits are handled
  outside, by make_env's StepCounter, so a timeout arrives as ``truncated`` and
  the critic bootstraps through it.  ``_armed`` guards the
  ~0.5% of resets that spawn the ball already inside the basin: success counts
  only once the ball has been outside at least once, otherwise such a sub-env
  emits a 1-step success on every reset until it happens to spawn outside.

* binary_terminal=False -- legacy: -1 per step while the ball is outside the
  basin, the native delta-distance shaping (pos_rew only, no collision term)
  inside, and ``done()`` suppressed so every episode runs the full horizon and
  the return reads -(steps spent off-goal).

DIVERGED ON PURPOSE from R2BC/src/scenarios/buzz_wire/wrappers/sparse_reward.py,
which still records the legacy schema.  The bundled demonstrations
(teachers/buzz_wire/demonstrations.pt) were recorded under it and are
reconciled offline by analysis/relabel_demos_binary.py into
demonstrations_binary.pt: the legacy recording marks the basin exactly
(reward != -1), which is what makes relabelling possible without re-collecting.
Do not "fix" the R2BC file to match.  Changing success_threshold still means
re-collecting, since the basin cannot be recomputed from saved observations.
Nor can collisions: the ball is not in the observation, so relabelled demo
episodes are cut at the basin, not at a preceding wall touch, and ~8% of the
teacher's successes would have been failures online.  Kept as a small,
one-directional optimism rather than re-collecting, which would also change
the teacher (user decision, 2026-09-19).
"""

from vmas.scenarios.buzz_wire import Scenario as BuzzWireScenario
from vmas.simulator.core import Agent
import torch


class SparseRewardBuzzWireScenario(BuzzWireScenario):
    def __init__(
        self,
        success_threshold: float = 0.1,
        out_penalty: float = -1.0,
        binary_terminal: bool = False,
    ):
        super().__init__()
        self.success_threshold = success_threshold
        self.out_penalty = out_penalty
        self.binary_terminal = binary_terminal

    def make_world(self, batch_dim, device, **kwargs):
        world = super().make_world(batch_dim, device, **kwargs)
        # torchrl builds info specs at env construction, before any reward()
        # call, so this must be a tensor from the start (never None).
        self.gt_reward = torch.zeros(batch_dim, device=device)
        self._armed = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        return world

    def reset_world_at(self, env_index=None):
        super().reset_world_at(env_index)
        if env_index is None:
            self._armed[:] = False
        else:
            self._armed[env_index] = False

    def _inside(self) -> torch.Tensor:
        dist = torch.linalg.vector_norm(
            self.ball.state.pos - self.goal.state.pos, dim=-1
        )
        return dist < self.success_threshold

    def _success(self) -> torch.Tensor:
        # Called by reward() (once per agent) and by done() every step; the
        # latch update is idempotent so the call count does not matter.
        inside = self._inside()
        self._armed |= ~inside
        return inside & self._armed

    def reward(self, agent: Agent):
        # Updates the cached self.pos_rew / self.collision_rew / self.rew on the
        # first agent's call; the base reward is shared across both agents.
        self.gt_reward = super().reward(agent)

        if self.binary_terminal:
            # self.collided is refreshed inside the base reward() on the first
            # agent's call, so here and in done() it is this step's contact state.
            return (self._success() & ~self.collided).to(self.pos_rew.dtype)

        inside_mask = self._inside().to(self.pos_rew.dtype)
        outside_mask = 1.0 - inside_mask
        # pos_rew only inside the radius (no collision term); flat penalty outside.
        return (self.pos_rew * inside_mask) + (self.out_penalty * outside_mask)

    def info(self, agent: Agent):
        info = super().info(agent)
        info["gt_reward"] = self.gt_reward
        return info

    def done(self):
        if self.binary_terminal:
            return self._success() | self.collided
        done = super().done()
        done.fill_(False)
        return done
