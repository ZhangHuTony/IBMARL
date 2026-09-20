"""
Binary terminal-reward navigation scenario.

Navigation's legacy sparse reward is a TorchRL transform
(src/environment/transforms/navigation_sparse.py), which runs after the
simulator has already decided ``done()`` and so can never end the episode where
the reward is paid.  This subclass moves both to scenario level, the way
buzz_wire's sparse variant already is, and is passed to torchrl's VmasEnv as a
scenario *instance* by make_env.resolve_scenario.

Schema (per agent, since navigation's reward has always been per agent):

* agent i is paid 1.0 exactly once, on the first step it is within its goal's
  radius (``goal.shape.radius`` = 0.05 -- VMAS's own ``on_goal`` predicate),
  and 0.0 on every other step;
* ``done()`` fires once every agent has been paid.  Episodes that never get
  there are truncated at ``horizon`` by make_env's StepCounter, and the critic
  bootstraps through that time limit.

``paid`` is the per-(env, agent) latch that makes the +1 pay once even if an
agent drifts off its goal and returns.  It is allocated in ``make_world`` and
cleared in ``reset_world_at`` rather than cached from ``reward()``: torchrl
calls ``done()`` on reset with rewards disabled, so a flag that is only
refreshed by ``reward()`` would report the *previous* episode's terminal on the
first step of the next one.  VMAS resets the world (and so this latch) before
querying dones, and partial resets arrive per sub-env through
``reset_world_at(env_index)``.

The recorded demonstrations were made under the legacy -1/step schema and are
relabelled offline to this one by analysis/relabel_demos_binary.py.
"""

from vmas.scenarios.navigation import Scenario as NavigationScenario
from vmas.simulator.core import Agent
import torch


class TerminalSuccessNavigationScenario(NavigationScenario):
    def make_world(self, batch_dim, device, **kwargs):
        world = super().make_world(batch_dim, device, **kwargs)
        n_agents = len(world.agents)
        self.paid = torch.zeros(batch_dim, n_agents, dtype=torch.bool, device=device)
        # torchrl builds info specs at env construction, before any reward()
        # call, so this must be a tensor from the start (never None).
        self.gt_reward = torch.zeros(batch_dim, n_agents, device=device)
        return world

    def reset_world_at(self, env_index=None):
        super().reset_world_at(env_index)
        if env_index is None:
            self.paid[:] = False
        else:
            self.paid[env_index] = False

    def reward(self, agent: Agent):
        # The base reward refreshes every agent's on_goal on the first agent's
        # call; its value (shaping + final bonus + collisions) is kept only as
        # the gt_reward diagnostic in info().
        gt_reward = super().reward(agent)
        i = self.world.agents.index(agent)
        self.gt_reward[:, i] = gt_reward

        newly_paid = agent.on_goal & ~self.paid[:, i]
        self.paid[:, i] |= agent.on_goal
        return newly_paid.to(self.pos_rew.dtype)

    def info(self, agent: Agent):
        info = super().info(agent)
        info["gt_reward"] = self.gt_reward[:, self.world.agents.index(agent)]
        return info

    def done(self):
        return self.paid.all(-1)
