# buzzwire2: what changed since paper3, and what the results say

Written 2026-08-26. Covers `results/buzzwire2` (16 runs, all succeeded) and how it
differs from `results/paper3`.

---

## 1. What this run is

paper3 is the sparse Navigation sweep from 2026-08-18/19. buzzwire2 is a sparse
Buzz-wire sweep finished 2026-08-26. It is not a re-run of paper3: the task is
different, several hyperparameters are different, and four code changes went in
between the two.

buzzwire2 contains 16 new runs:

| variant | seeds |
| --- | --- |
| `ibmarl_strict` (the paper's IBMARL) | 0–4 |
| `ibmarl` (per-agent mixing) | 0–4 |
| `ibmarl_strict_hard` (w/o soft selection) | 0–2 |
| `ibmarl_strict_1critic` (w/o critic ensemble) | 0–2 |

The baselines (`maddpg`, `rlfd`, `rft`, `bc_eval`) are **symlinks to
`results/buzzwire1`**, not new runs. None of them has an arbiter, so the arbiter
fix below cannot affect them. They were, however, measured under the older
evaluation code path — see §7.

---

## 2. Results

Final return = mean of a seed's last 10 evaluations, then mean ± SD across seeds.
Teacher level as measured by `bc_eval` on this task: **−86.4**.

| variant | n | RL+IL (executed) | RL actor only | seeds reaching teacher |
| --- | --- | --- | --- | --- |
| IBMARL | 5 | −87.2 ± 8.5 | −92.9 ± 10.2 | **5/5** |
| w/ per-agent mixing | 5 | **−79.3 ± 8.8** | −104.1 ± 43.1 | 3/5 |
| w/o soft selection | 3 | −85.2 ± 16.9 | −118.7 ± 60.7 | 1/3 |
| w/o critic ensemble | 3 | −82.4 ± 4.5 | **−89.4 ± 11.0** | 2/3 |
| RLfD | 5 | — | −122.9 ± 45.1 | 4/5 |
| RFT | 3 | — | −126.5 ± 2.2 | 0/3 |
| MADDPG | 5 | — | −194.6 ± 5.7 | 0/5 |

Median steps to teacher, RL-actor protocol: IBMARL 456k (5/5), RLfD 568k (4/5).

### Against buzzwire1, same variant, same seeds

| | buzzwire1 | buzzwire2 |
| --- | --- | --- |
| `ibmarl_strict` final return | −126.0 | −87.2 |
| SD across seeds | 67.5 | 8.5 |
| seeds stuck at −200 | 2/5 | 0/5 |
| per-seed | −74, **−200**, −68, **−200**, −89 | −100, −83, −83, −91, −79 |

This is the main thing the run was for, and it worked. The two dead seeds are
gone and the spread drops by a factor of eight.

It cost something. buzzwire1's three surviving seeds averaged about −77;
buzzwire2 averages −87. Removing the failure mode also removed the best case.

---

## 3. Change 1: the task

paper3 is Navigation, 3 agents. buzzwire2 is Buzz-wire, 2 agents. Different
scenario, different teacher checkpoint, different reward radius, different
horizon. Nothing below should be read as a controlled comparison against paper3's
numbers.

---

## 4. Change 2: hyperparameters

From the saved `config.yaml` of `ibmarl_strict/seed_0` in each sweep.

| key | paper3 | buzzwire2 | why |
| --- | --- | --- | --- |
| `scenario_name` | navigation | buzz_wire | |
| `n_agents` | 3 | 2 | task |
| `horizon` | 100 | 200 | task |
| `gt_radius` | 0.4 | 0.1 | sparse basin; 10× the native success test (0.01), inside the corridor half-width (0.125) |
| `frames_per_batch` | 1000 | 8000 | 40 sub-envs × 200 steps. A batched VMAS step costs the same at 10, 20, 40 or 80 sub-envs (67 ms measured), so this is nearly free wall-time |
| `n_iters` | 300 | 75 | with the above, same 4× fewer sequential batches |
| `total_frames` | 300k | 600k | buzz-wire needs the longer budget |
| `training.n_optimiser_steps` | 500 | 2000 | keeps updates-per-frame at 0.25; both sweeps do 150,000 gradient steps total |
| `eval_interval` | 1 (default) | 4 | evaluation is ~64% of an iteration here (it rolls out twice the collection horizon). 20 evaluations over the run |
| `ibmarl_replay_size` | 200k | 400k | the ring wraps at 400k = iteration 50 of 75, matching navigation's 67% demo survival |
| `ibmarl_min_warm_up_frames` | 5000 | 40000 | 5 iterations = 200 teacher episodes = 10% of the buffer |
| `ibmarl_warm_up_policy` | — | `teacher` | **new key**, see §5.4 |
| `eval_rl_only` | — | `true` | **new key**, see §5.2 |
| `eval_critic_reduction` | — | `min_all` | **new key**, see §5.2 |
| `r2bc_checkpoint_path` | navigation, 2026-01-29 | buzzwire, 2026-08-24 | 16 demonstrations, 3,200 transitions |

Unchanged: `strict`, `soft`, `temperature` (0.05), `num_critics` (3),
exploration noise (sigma 0.1, flat), learning rates, `train_batch_size` (1024),
`gamma`, `polyak_tau` (0.001).

**buzzwire1 → buzzwire2** is a much smaller delta. Only four keys differ:
`ibmarl_min_warm_up_frames` 5000 → 40000, plus the three new keys above.

---

## 5. Change 3: code

### 5.1 The collector was running a stale copy of the arbiter

The bug. `AdditiveGaussianModule` registers its sigma/mean/std buffers on the
CPU. `SyncDataCollector` casts a policy to its device by deep-copying it whenever
any parameter or buffer sits elsewhere. `ActionArbiter` is a plain attribute, not
a registered submodule, so torchrl's weight mapping does not see it — the copy
got its own arbiter, and that arbiter's critics stayed at their initial random
weights for the whole run.

This affected **every IBMARL run to date, paper3 and buzzwire1 included**.

What was still correct before the fix, verified: the TD target
(`bootstrap_proposal` reads the experiment's own arbiter), the critic update, the
actor update (the copy's actor weights share storage with the original),
evaluation, and demonstration preloading. Only the behaviour policy's *selector*
was stale. In practice collection mixed teacher and RL actions at a fixed random
rate that was drawn once per seed and never moved.

Fix, in three parts:

- `ibmarl/modules.py` — move the noise module onto the policy's device, so
  torchrl has no reason to copy.
- `ibmarl/data.py` — after building the collector, raise if
  `collector.policy is not exploration_policies`. If this ever recurs the run
  dies instead of quietly producing wrong numbers.
- `ibmarl/arbiter.py` — `__deepcopy__` returns `self`.

Fingerprint on disk: buzzwire1 seed 1 has `mean_q_diff` pinned at 0.12–0.19 and
`var_q_diff` at 0.000 while `critic_loss` climbs 0.035 → 13.1 — the critics were
learning, the acting arbiter never noticed. buzzwire2 starts at `mean_q_diff`
2.31 and decays to 0.06.

### 5.2 Evaluation is now the RL+IL policy

Previously `eval_reward_mean` measured the RL actors alone, and
`rl_only_episode_reward_mean` was assigned the same value — a duplicate, not a
second measurement.

Now:

- `eval_reward_mean` is the arbitrated RL+IL policy: greedy argmax, no
  exploration noise, minimum over all three target critics. This is the policy
  IBMARL actually executes, and what IBRL reports.
- `rl_only_episode_reward_mean` is a real second evaluation of the RL actors.
- New columns `eval_rl_action_fraction` and `eval_protocol`.

Both protocols run on the same episodes at the same iterations, so their
difference is a paired comparison. Checkpoints now also save
`critic_checkpoint.pt` so a run can be re-evaluated after the fact without
retraining.

### 5.3 VMAS random-number guard

`vmas.simulator.environment.Environment` keeps `vmas_random_state` as a single
class-level list. The training environment, the evaluation environment and the
rendering environment all draw from it. Evaluation rollouts were therefore
shifting the resets the collector saw next — evaluating changed what was trained
on.

`vmas_rng_guard()` snapshots that list and restores it in place. Adding a second
evaluation protocol would otherwise have doubled the disturbance.

### 5.4 Teacher warm-up (buzz-wire only)

During warm-up the teacher's proposal is now executed unconditionally, so the
arbiter's first decision is made by a critic that has already seen the teacher
succeed. Navigation keeps the old behaviour (`ibmarl_warm_up_policy: arbiter`).

The reason: on buzz-wire the RL actor alone never reaches the goal (MADDPG 0/5),
so letting an untrained critic arbitrate the first batches decided the run by
coin flip. In buzzwire1 the two seeds that opened at 91% and 43% RL actions sat
at −200 for all 75 iterations; all three seeds that opened below 30% RL crossed
the teacher.

The teacher is robust to the exploration noise it carries during arbitration —
clean −84.4 with 97.5% of episodes reaching the goal, at sigma 0.1 −84.3 and
98.8% — so its proposals are executed with that noise rather than noise-free.
Uniform random actions reach the goal in 2.5% of episodes, for scale.

### 5.5 Already present in buzzwire1, absent from paper3

Not new tonight, but they separate paper3 from both buzz-wire sweeps:

- `load_config` gained a per-scenario experiment layer
  (`config/experiments/<exp>_<scenario>.yaml`), so a second scenario can supply
  its own teacher paths without touching the navigation defaults.
- `--sparse` / `--dense` command-line flags.
- `should_evaluate(iteration)` and `eval_interval`, wired through MADDPG, RLfD
  and RFT.
- `SparseRewardBuzzWireScenario`.

---

## 6. Change 4: analysis scripts

- `paper3_figures.py`: `--protocol rl|executed`, protocol-suffixed output names,
  axis preset for buzzwire2, and missing variants are skipped rather than
  crashing.
- `fig4_replication` had been broken since the ablations were renamed (KeyError
  on `ibmarl_strict_hard`); it last produced output on 2026-08-18. Fixed. It is
  navigation-only and is not part of the buzzwire2 figure set.
- `fig_mixing_comparison.py`, `steps_to_teacher.py`: `--protocol` flag.
- `eval_checkpoints.py`: `--protocol rl|rl_il|both`.

Figures are in `analysis/figures/buzzwire2/`. Files without a suffix are the
RL-actor protocol; `*_executed.*` are the RL+IL protocol.

---

## 7. Reading the results honestly

**The stabilisation worked.** Spread across seeds fell from 67.5 to 8.5 and no
seed failed. IBMARL is the only method with 5/5 seeds reaching the teacher.

**Both IBMARL variants beat every baseline** by 35–115 return, and beat them on
consistency too: SD 8.5–8.8 against RLfD's 45.1.

**The ablations do not separate on this task.** Under the RL+IL protocol IBMARL
(−87.2) is last among its own four configurations, behind per-agent mixing
(−79.3), single-critic (−82.4) and argmax selection (−85.2). Under the RL-actor
protocol it beats two of the three but loses to single-critic (−89.4). With three
seeds and spreads of 4.5 to 17, none of these gaps means anything. The one axis
where IBMARL does separate is reliability: 5/5 seeds reach the teacher, against
1/3 without soft selection and 2/3 without the critic ensemble.

**The strict-versus-mixing ordering flips depending on protocol.** Mixing wins on
the executed policy (−79.3 vs −87.2) and loses on the RL actor (−104.1 vs −92.9,
SD 43.1 vs 10.2). `ibmarl/seed_1` shows why: RL+IL −88.5 but RL-actor −178.6. Per
agent mixing can take the teacher's action for the agent whose RL actor is broken
and keep the RL action for the other; the all-IL-or-all-RL choice cannot. Mixing
buys executed performance by leaning on the teacher, not by learning a better
policy.

This contradicts paper3, where strict beat mixing (crossing median 43k vs 80k on
navigation). It is a real result, not an artefact. But it means the claim
"IBMARL is the strict variant, mixing is the worse ablation" does not hold on
buzz-wire under the RL+IL protocol that is now the headline.

---

## 8. Open items

- **paper3's navigation numbers were produced with the frozen acting arbiter.**
  That sweep needs re-running before the paper can describe the method that
  produced its table. This is the largest outstanding item.
- The buzz-wire baselines are buzzwire1 runs. They have no arbiter, so §5.1,
  §5.2 and §5.4 do not touch them, but they were measured before the RNG guard
  in §5.3.
- Two teacher levels are in circulation: −80.6 (reported by R2BC on its own
  evaluation) and −86.4 (`bc_eval` on this sweep). The figures use −86.4. The gap
  is partly explained by the next item.
- `bc_eval_experiment.py:70` rolls out `self.env` (seed S) while `evaluate()`
  uses `self.eval_env` (seed S+10,000), so the teacher reference is measured on a
  different sequence of episodes than the curves it is compared against. Not
  fixed, because fixing it moves a published number.
- RFT has 3 of 5 seeds on buzz-wire. Seeds 3 and 4 were never run.
- `ibmarl_strict/seed_3` ends at −133.3 after nine evaluations in the −64 to −95
  band. Report final-10 means, not last-evaluation values.
- The ablations have 3 seeds. If the ablation figure is going into the paper, 5
  seeds would make the reliability claim in §7 defensible; at 3 seeds "1/3 versus
  5/5" is thin.
