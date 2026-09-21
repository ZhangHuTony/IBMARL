# IBMARL

Multi-agent extension of IBRL: a frozen imitation policy trained with R2BC helps
an off-policy MARL learner explore and bootstrap on sparse-reward VMAS tasks.
This glossary fixes the canonical vocabulary used in the code, the analysis, and
the workshop paper.

## Language

**Teacher**:
The frozen decentralized R2BC imitation policy loaded from a checkpoint; it
proposes actions but is never updated.
_Avoid_: BC policy, IL policy (as a name; fine as a descriptor), expert

**Arbiter**:
The critic-scored selector that chooses between the teacher's joint action and
the RL joint action by scoring both and Boltzmann-sampling over the two scores.
_Avoid_: action selector, mixer

**Proposal**:
A candidate action offered to the arbiter — one from the teacher and one from
the RL actor per agent, both perturbed with the same exploration noise.

**Bootstrap proposal**:
The arbiter applied at the next observation to target actors and teacher, to
pick the action whose value the TD(0) target bootstraps from.

**Gate**:
The arbiter's preference for the teacher's proposal at a training observation,
recomputed from the current target critics.

**Gated imitation term**:
An actor-loss term that pulls the RL actor's action toward the teacher's
proposal in proportion to the gate.
_Avoid_: BC regularizer (that is RFT's demo-action term), distillation loss

**Uniform imitation term** (ablation):
The gated imitation term with the gate held open.

**Per-agent mixing** (ablation):
A variant expanding the arbiter's candidate set to all 2^N teacher/RL
combinations, mixing the two policies within one team decision. Measured to
slow early learning (crossing median 80k vs 43k); not part of IBMARL proper.
_Avoid_: combinatorial mode

**Strict** (code flag, not a paper term):
`strict: True` in the config selects IBMARL's two-candidate joint arbiter.
NAMING MAP: the paper's "IBMARL" = the `ibmarl_strict` runs on disk; the
`ibmarl` runs = "w/ per-agent mixing"; `ibmarl_strict_hard` / `ibmarl_strict_1critic`
= the paper's "w/o soft selection" / "w/o critic ensemble" (one-factor ablations).
The old `ibmarl_hard` / `ibmarl_1critic` runs (mixing base) are not in the paper.

**Soft / Hard** (ablation axis):
Soft = Boltzmann-sample the executed candidate at temperature 0.05;
Hard = argmax.

**Critic ensemble**:
E=3 centralized critics per agent, all trained to the same target; the arbiter
scores with the min over 2 randomly sampled target members (REDQ-style
pessimism). Single-critic (E=1) is the ablation.

**Teacher level**:
The frozen teacher's evaluation return (−48.7 on sparse Navigation); the
reference line for sample-efficiency claims.
_Avoid_: BC level, R2BC level

**Steps to teacher**:
Sample-efficiency metric: environment steps until the 9-evaluation-smoothed
RL-actor evaluation return first exceeds the teacher level (median over seeds).
Read off the RL-actor protocol, not the arbitrated one, which starts near
teacher level by construction.

**Warm-up**:
The opening online steps collected before any gradient update (5k on
navigation, 40k on buzz-wire).

**Teacher warm-up**:
A warm-up in which the teacher's proposals are executed unconditionally, so
the arbiter's first decision is informed by teacher successes rather than by
an untrained critic. IBRL's warm-up; used on buzz-wire, where the RL actor
alone never reaches the goal and an arbitrated warm-up decides the run by
coin flip.
_Avoid_: pre-training, BC warm-start (nothing is trained during it)

**Demo dilution**:
IBRL-style single replay buffer pre-loaded with demonstrations that online
transitions gradually crowd out; no separate demo buffer, no mixing schedule.
_Avoid_: demo annealing (that is RLfD's mechanism, not IBMARL's)

**Sparse Navigation**:
VMAS Navigation with the per-agent reward overwritten to −1 whenever the agent
is ≥ 0.4 from its goal center, and the native (near-zero) VMAS reward inside.
Returns read as −(steps spent off-goal).

**Arbitrated evaluation**:
The primary evaluation protocol: the arbiter choosing between the teacher's and
the RL actors' proposals, greedily and without exploration noise, on a dedicated
fixed-seed environment. The policy IBMARL actually executes, and what IBRL
reports.
_Avoid_: combined-policy evaluation, IBMARL evaluation (RL+IL is fine as a
descriptor)

**RL-actor evaluation**:
The secondary evaluation protocol: deterministic RL actors only — the teacher
and arbiter are not in the loop — measured on the same eval episodes as the
arbitrated evaluation, so the two are paired. The protocol every baseline is
measured under, and the one the teacher level is compared against.
_Avoid_: RL-only reward (that is a column name, not the term)
