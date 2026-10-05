"""
Balance with the episode end suppressed: every episode runs the full horizon.

vmas 1.5.2's balance ends an episode when the line or package touches the floor
or the package overlaps the goal.  Under the gt-in-radius sparse schema
(transforms/balance_sparse.py: -1 per step outside the basin) that terminal is
a suicide incentive -- dropping the package at step 5 costs -5 against -300 for
carrying it off-goal for the whole horizon -- and it is not the MDP the bundled
demonstrations were recorded in: teachers/balance/demonstrations.pt holds 36
episodes of exactly 300 steps, with the recorder's on-goal flag going True
mid-episode and the episode continuing.  Suppressing ``done()`` here makes the
online MDP match the recording, as the legacy branch of
scenarios/buzz_wire_sparse.py does for buzz-wire.

Only ``done()`` is overridden.  The reward stays with the TorchRL transform so
the online predicate and analysis/relabel_demos_sparse.py read literally the
same observation slice; ``on_the_ground`` is refreshed inside the base
``reward()``, not ``done()``, so nothing else depends on this method.  The time
limit is VmasEnv's ``max_steps=horizon`` (make_env's non-binary path), which
VMAS merges into the done flag the collector sees, so ``("next", "done")`` is
True exactly at t == horizon-1 and one rollout of ``horizon`` steps is exactly
one episode per sub-env.

Selected by make_env.resolve_scenario when ``scenario_name == "balance"``,
``sparse_rewards`` and ``suppress_done`` (config/environments/balance.yaml,
default True).  Named apart from ``balance_sparse.py``, which the
``binary_balance_exp`` branch uses for its 0/1 terminal scenario.
"""

import torch
from vmas.scenarios.balance import Scenario as BalanceScenario


class FixedHorizonBalanceScenario(BalanceScenario):
    def done(self):
        return torch.zeros(
            self.world.batch_dim, dtype=torch.bool, device=self.world.device
        )
