# IBMARL — Implementation Report (methods-section source material)

**Purpose.** This document describes exactly what each algorithm in this repository *does in code*, at the
revision the reported results were produced from. It is written to be pasted into a chat that already has
the paper's framing, so it prioritises mechanism, exact hyperparameters, and honest flagging of
implementation/algorithm mismatches over narrative.

---

## 0. Provenance and the assumption this report is written under

**Assumed code state: commit `e0da62b` ("SparseReady", 2026-03-11) — the tip of `main` immediately before
the post-ConnorWorking-merge changes.**

Per the instruction that the results predate any change made after the ConnorWorking merge, everything below
describes `e0da62b` and *excludes*:

| Excluded | What it is |
| --- | --- |
| `42dc2e5` (2026-08-15) | "Fix demo-loader self-transitions, critic ensemble, noise annealing" |
| `c7a6157` (2026-08-16) | before/after bug-fix comparison analysis |
| current uncommitted working tree | rewritten `evaluate()`, checkpoint/resume mixin, shared video recorder, MADDPG/IBMARL refactors |

Sanity notes on provenance, in case they matter for the paper:

* `backup/main-pre-reset` (the pre-force-push `main`) differs from `e0da62b` only in four cosmetic/plumbing
  spots (import moves, a `make_env(num_envs, seed)` override, two extra calls in `run_experiment`). **No
  algorithmic difference.** So the report holds for either lineage.
* Every `results/*/data/metrics.csv` in the repo has the header
  `iteration,group,episode_reward_mean[,rl_action_fraction]` — i.e. **the learning curves you have are the
  in-batch behaviour-policy return, not a separate deterministic evaluation.** See §2.7; this matters for
  how the y-axis is described.
* The run `config.yaml` files under `results/` (e.g. `ibmarl_navigation_2026-08-12_11-31-10`) lack the
  `exploration_noise`, `ibmarl_replay_size` and `ibmarl_min_warm_up_frames` keys. Those runs therefore used
  the in-code defaults: σ = 0.1 constant, buffer capacity = `memory_size` = 1e6, warm-up = 5000 frames. This
  changes nothing qualitatively (§5.5).

---

## 1. Task and environment

| Item | Value |
| --- | --- |
| Simulator | VMAS 1.5.2 via TorchRL `VmasEnv` (`torchrl` 0.10, `tensordict` 0.10, torch 2.9.1+cu128) |
| Scenario | `navigation` |
| Agents | 3, one group (`"agents"`), continuous actions |
| Observation dim | 18 per agent |
| Action dim | 2 per agent, box in [−1, 1] |
| Episode horizon | 100 steps (`max_steps=100`) |
| Vectorisation | `frames_per_batch // horizon` = 1000 // 100 = **10 parallel sub-environments** |
| Device | CUDA if available, else CPU |

`src/environment/make_env.py` wraps the base env as
`TransformedEnv(VmasEnv) → NavigationSparseReward → RewardSum`. Because the sparse transform is appended
*before* `RewardSum`, the logged `episode_reward` is the sum of the **transformed** reward.

### 1.1 Sparse reward transform (`src/environment/transforms/navigation_sparse.py`)

This is **not** a 0/1 success reward (that variant exists but is commented out). What runs is:

```
d_i      = || obs_i[4:6] ||                 # relative goal vector, per agent
r_i      = r_i^VMAS      if d_i < gt_radius
         = -1            otherwise
gt_radius = 0.4   (navigation);  1.05 (balance, using obs[8:10] of agent 0)
```

So the agent receives the ordinary VMAS shaped reward only inside a 0.4-radius ball of its goal, and a flat
−1 penalty everywhere else. Consequences worth stating in the paper:

* Per-agent episode return lies in roughly **[−100, +8]**; a policy that never reaches the goal scores −100.
* Outside the ball the reward carries **no gradient information** — this is what makes the task
  exploration-hard and is the reason a demonstration prior helps.
* Empirically, the in-ball reward is small: across the 2400 recorded demo transitions the reward range is
  `[-1.0, +0.079]`.

`transform_reward_spec` re-declares the group reward spec as unbounded so `check_env_specs` passes.

---

## 2. Shared substrate (identical across MADDPG / RLfD / RFT / IBMARL unless stated)

### 2.1 Actor

`MultiAgentMLP`, **decentralised** (`centralized=False`) and **not parameter-shared**
(`share_params=False`): each of the 3 agents has its own network mapping its own 18-dim observation to a
2-dim action. Depth 2, 256 units/layer, `Tanh` activation. Wrapped in a TorchRL `ProbabilisticActor` with a
`TanhDelta` distribution bounded by the action spec — i.e. **deterministic** policy, squashed into the
action box. `return_log_prob=False`.

### 2.2 Critic

`MultiAgentMLP`, **centralised** (`centralized=True`) and **not parameter-shared**: the per-agent input is
`concat(obs_i, a_i)` (18 + 2 = 20), and because the MLP is centralised each agent's critic head consumes the
concatenation across all 3 agents (60-dim) and emits **one scalar Q per agent**. Depth 2, 256 units, `Tanh`.
This is the standard CTDE/MADDPG arrangement: decentralised execution, centralised per-agent critics.

### 2.3 Data collection

`SyncDataCollector` with `frames_per_batch=1000`, `total_frames = 1000 × n_iters`. One iteration collects
exactly **one full episode in each of the 10 sub-environments** (10 × 100 = 1000 team frames).

Exploration is `AdditiveGaussianModule` on the action key with
`sigma_init = sigma_end = 0.1` (constant σ = 0.1 Gaussian noise; the annealing schedule
`annealing_num_steps = total_frames // 2` is therefore inert). MADDPG/RLfD/RFT hard-code σ = 0.1 in
`maddpg.py`; IBMARL reads `exploration_noise.{sigma_init,sigma_end}` from the config (default 0.1/0.1) and
also exposes `--sigma_init/--sigma_end` CLI overrides. The noise module respects TorchRL's
`ExplorationType`, so evaluation/rendering under `ExplorationType.MODE` is noise-free.

### 2.4 Replay

`ReplayBuffer(LazyMemmapStorage(memory_size=1e6), RandomSampler(), batch_size=1024)`, one per group, with a
`.to(device)` transform when on GPU. IBMARL differs (§5.5).

### 2.5 Optimisation loop (per training iteration)

```
collect 1000 frames with the behaviour policy
push them into the replay buffer
repeat 500 times:                    # training.n_optimiser_steps
    sample 1024 transitions
    critic step  (Adam, lr 1e-4, grad-norm clip 1.0)
    actor  step  (Adam, lr 1e-4, grad-norm clip 1.0)
    polyak update of target nets, tau = 0.001
```

With `n_iters = 250` this is **125 000 gradient steps and 125 000 target updates per run**, at a
replay-ratio of 500 updates per 1000 environment frames (0.5 updates/frame, batch 1024).

### 2.6 Hyperparameters (`config/base.yaml` + `config/environments/navigation.yaml`)

| Parameter | Value |
| --- | --- |
| `n_iters` | 250 (`total_frames` = 250 000) |
| `frames_per_batch` | 1000 |
| `train_batch_size` | 1024 |
| `n_optimiser_steps` | 500 |
| `lr` (Adam, actor and critic) | 1e-4 |
| `max_grad_norm` | 1.0 |
| `gamma` | 0.99 |
| `polyak_tau` | 0.001 |
| `memory_size` | 1 000 000 |
| exploration σ | 0.1, constant |
| `n_agents` / `horizon` / `gt_radius` | 3 / 100 / 0.4 |
| **unused keys** | `num_critic_updates: 3`, `checkpoint_interval: 10` — never read anywhere in `src/` |

**Do not cite `num_critic_updates: 3` as a critic-update ratio; it is dead config.**

### 2.7 Metrics and evaluation — read this before writing the results section

`src/util/metrics_logger.py` writes a CSV whose columns are the union of whatever each experiment logs.

* **`episode_reward_mean`** (the column your result files actually contain) = the mean, over all per-agent
  episode returns that terminated inside the collected batch, of `("next", group, "episode_reward")`. It is
  measured on the **behaviour policy**: for MADDPG/RLfD/RFT that is the actor *plus σ = 0.1 exploration
  noise*; for IBMARL it is the *arbitrated IL/RL mixture with noise on both candidates* — i.e. **not** the RL
  policy alone. Each terminated episode contributes 3 samples (one per agent).
* **`rl_action_fraction`** (IBMARL only) = mean of the arbiter's per-agent binary choice mask over the batch;
  the fraction of agent-timesteps in which the RL candidate was executed rather than the IL candidate.
* **`eval_reward_mean`** is emitted by the `e0da62b` code but is **broken**, and is absent from every
  `results/*/metrics.csv` in the repo. In `BaseMARLExperiment.evaluate()` the code does
  `out.get(("next", group, "done"))` on a raw `env.rollout(...)` output. The per-group `done` key is only
  created by `process_batch` on *collector* batches, so on a rollout this returns `None` (verified against
  tensordict 0.10); `episode_rewards[None]` then inserts an axis instead of masking, `n = min(20, 1) = 1`,
  and the logged number is the mean of the *running* episode return over **every timestep** of the rollout
  rather than the return at episode end. For a near-linearly accumulating return that is ≈ **half** the true
  return. If any curve in the paper came from an `eval_reward_mean` column, its scale is wrong and it must
  not be plotted alongside post-fix numbers.
* IBMARL also logs `rl_only_episode_reward_mean`, which is **assigned the same value as `eval_reward_mean`**
  — it is not an independent RL-only measurement despite the name.
* Also logged: `elapsed_time`, `episode`, `step`, `train_step`, `speed_fps`, `actor_loss`, `critic_loss`,
  `replay_size`; plus `demo_fraction` (RLfD), `bc_loss`/`bc_weight` (RFT),
  `mean_action_diff`/`mean_q_diff`/`var_q_diff` (IBMARL).

### 2.8 Seeding

`BaseMARLExperiment._setup_seed` sets `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`;
`make_env` passes the same seed to `VmasEnv`. `run_experiment.py` takes `--seeds N --seed-start k` and runs
seeds `k … k+N−1` sequentially, each in its own timestamped `results/<exp>_<scenario>_<timestamp>/` directory
with the fully merged `config.yaml` written out. Note the entry point does **not** accept `--seed`
(singular).

### 2.9 Config merging

`load_config(scenario, exp_type)` loads `config/base.yaml`, then `config/environments/<scenario>.yaml`
(with `training` merged key-wise, everything else `dict.update`), computes
`total_frames = frames_per_batch × n_iters`, then `dict.update`s `config/experiments/<exp_type>.yaml`. The
`extends:` field in the YAMLs is decorative — the merge order is hard-coded.

---

## 3. MADDPG (baseline) — `src/experiments/maddpg.py`

Standard TorchRL multi-agent DDPG; no demonstrations anywhere.

**Loss.** One `torchrl.objectives.DDPGLoss` per group with `delay_value=True`, TD(0) value estimator,
`gamma = 0.99`, keys remapped onto the group's nested `state_action_value` / `reward` / `done` /
`terminated`. Reading TorchRL 0.10's implementation:

```
critic:  y      = r + gamma * (1 - terminated) * Q_target(s', pi(s'))
         L_Q    = MSE( Q(s, a), y )
actor:   L_pi   = - mean_over(batch, agents) Q_detached(s, pi(s))
```

Two details that are easy to state wrongly:

* `delay_actor` is left at its default `False`, so **the bootstrap action comes from the *online* actor**
  (detached), not a target actor. Only the critic is delayed.
* `loss_actor` is computed against `value_network_params.detach()`, so the actor backward pass does **not**
  contaminate critic gradients even though the loop calls `.backward()` on both losses before either
  `zero_grad()`.

**Update.** Both losses come from one `loss_module(minibatch)` call; the loop then does, per optimiser step,
`backward → clip(1.0) → step → zero_grad` for `loss_actor` and then for `loss_value`, followed by
`SoftUpdate(tau=0.001).step()`. The exploration σ scheduler is stepped once per iteration.

**Termination handling.** `process_batch` expands the global `("next","done")` / `("next","terminated")` to
per-agent shape so the loss sees masks aligned with the per-agent reward. Under a fixed 100-step horizon the
env emits time-limit `done` with `terminated = False`, so the value target bootstraps across episode
boundaries — correct for a time-limited task.

---

## 4. Baselines that use demonstrations

### 4.0 The demonstration data

Both demo baselines and IBMARL consume the same `demonstrations.pt` produced by the sibling R2BC project:

| Field | Shape | Notes |
| --- | --- | --- |
| `obs` | (2400, 3, 18) | 24 demonstrations × 100 steps |
| `act` | (2400, 3, 2) | already clipped to [−1, 1] |
| `rewards` | (2400, 3) | range [−1.0, +0.079] → **already the sparse-transformed reward** |
| `next_obs` | (2400, 3, 18) | **present in the file but never read at `e0da62b`** (see §6.1) |
| `dones` | (2400,) | all `False` |

The `_load_demonstrations_into_buffer` helper is duplicated verbatim in `rlfd_experiment.py`,
`rft_experiment.py` and `ibmarl/data.py`. It builds a TensorDict with `(group, observation/action/
episode_reward)`, and constructs the `"next"` subtree as **`agents_data.clone()`** with `reward` set from
`rewards` and `done`/`terminated` all-False. §6.1 explains why that matters.

### 4.1 RLfD — `src/experiments/rlfd_experiment.py`

`RlfdExperiment(MaddpgExperiment)`. **The only change from MADDPG is where the minibatch comes from.** Same
actor, same critic, same `DDPGLoss`, same optimiser schedule, same exploration.

Two buffers per group: a demo buffer pre-loaded with the 2400 demo transitions, and an online buffer fed by
the collector. Once per iteration:

```
progress   = min(1, iteration / (n_iters - 1))
demo_frac  = 0.5 * (1 - progress)                # 0.50 at iter 0 -> 0.0 at the last iter
demo_bs    = round(1024 * demo_frac)             # clamped to [0, 1023]
online_bs  = 1024 - demo_bs
```

and every one of the 500 optimiser steps in that iteration samples `demo_bs` demo transitions and
`online_bs` online transitions and concatenates them (`_concat_minibatches`, key-wise `torch.cat` over the
intersection of keys). This is the classic linearly-annealed demonstration-oversampling recipe: 50 %
demonstrations at the start, 0 % at the end, with no separate BC term and no prioritisation.

### 4.2 RFT — `src/experiments/rft_experiment.py`

`RftExperiment(MaddpgExperiment)`. **The only change from MADDPG is an added BC regulariser on the actor.**
RL data flows through the ordinary single online buffer exactly as in MADDPG; the demo buffer is used
*solely* for the BC term.

```
L_actor^RFT(t) = L_actor^MADDPG  +  alpha * lambda(t) * L_BC
alpha          = 0.1                               # rft.bc_alpha
lambda(t)      = max(0, 1 - t / 150)               # rft.bc_anneal_n_iters = 150, t = iteration index
L_BC           = MSE( pi(s_demo), a_demo )         # fresh 1024-sample demo minibatch, every optimiser step
```

`lambda` is recomputed once per iteration and held fixed across that iteration's 500 steps; `L_BC` is drawn
fresh each step, so a run performs 500 BC minibatches per iteration for the first 150 iterations and none
thereafter. The critic loss is untouched. Note for the metrics table: the logged `actor_loss` is captured
**before** the BC term is added (the BC part is reported separately as `bc_loss`/`bc_weight`).

### 4.3 BC eval — `src/experiments/bc_eval_experiment.py`

No learning. Loads the R2BC checkpoint, wraps `get_action` per group into a `TensorDictModule`, rolls out
`horizon × (n_episodes + 5)` steps under `ExplorationType.DETERMINISTIC`, and reports the mean of the first
20 completed per-agent episode returns. Used to place the imitation prior on the reward axis.

### 4.4 The imitation policy (`DecentralizedMiniBC`, `src/r2bc/mabc.py`)

The frozen prior is deliberately tiny and fully decentralised: **per agent, `Linear(18→8) → ReLU →
Linear(8→2)`** (`hidden_size=8`, `hidden_layers=1` in the checkpoint), three such heads, no parameter
sharing. `R2bcPolicy.get_action` flattens `[B, N, 18] → [B, 54]`, the module slices the flat vector back into
per-agent chunks, and the output is reshaped to `[B, N, 2]` and **clamped to [−1, 1]**. It is loaded with
`.eval()` and never receives gradients. A shape smoke-test runs at construction.

---

## 5. IBMARL — `src/experiments/ibmarl_experiment.py` + `src/experiments/ibmarl/`

IBMARL is the multi-agent adaptation of IBRL: a frozen imitation policy and a learned RL policy each propose
an action, and a **critic-scored arbiter** decides, per timestep, which proposal(s) to execute — and, in the
TD target, which proposal to bootstrap from. The multi-agent extension is that the arbiter selects over
**joint** actions built from per-agent choices between the two proposals.

Config as run: `strict: False`, `soft: True`, `temperature: 0.05`, `num_critics: 3`,
`ibmarl_replay_size: 200000`, `ibmarl_min_warm_up_frames: 5000`, `exploration_noise: {0.1, 0.1}`.

### 5.1 Networks (`ibmarl/networks.py`)

* IL policy: frozen `DecentralizedMiniBC` (§4.4).
* RL policy: identical construction to the MADDPG actor (§2.1).
* Critics: an `nn.ModuleList` of **`num_critics = 3`** independently initialised members, each with the exact
  MADDPG critic architecture (§2.2).
* Targets: `copy.deepcopy` of the actor and of the whole critic ensemble, `.eval()`, `requires_grad_(False)`.

### 5.2 Behaviour policy / acting (`ibmarl/data.py`, `ibmarl/modules.py`)

Per group the collector policy is

```
RL actor  ->  AdditiveGaussian(sigma=0.1)  ->  OverWriteActionWithBestComb(arbiter)
```

`OverWriteActionWithBestComb` is a `@torch.no_grad()` module that reads `(group, observation)` and the noisy
RL action out of the tensordict, calls the arbiter, and **overwrites** `(group, action)` with the arbitrated
joint action, additionally writing `arbiter_choice` and the three diagnostic scalars back into the batch so
they survive into the collected data.

### 5.3 The arbiter, acting mode (`ibmarl/arbiter.py: _best_act_comb`, the `strict: False` path)

```
a_IL = clamp(pi_IL(s), -1, 1)                      # frozen prior
a_IL = a_IL + N(0, 0.1)                            # SAME exploration noise as the RL branch
a_RL = pi_RL(s) + N(0, 0.1)                        # already noised upstream

choices = {0,1}^N                                  # N = 3  ->  K = 2^3 = 8 joint candidates
joint[k][i] = a_IL[i] if choices[k][i]==0 else a_RL[i]

idx      = 2 distinct indices sampled uniformly from {0..num_critics-1}
Q[j,k,i] = Q_target^(idx[j]) (s, joint[k])         # TARGET critics, batched over K
Qmin[k,i]= min_j Q[j,k,i]                          # per-agent pessimistic (clipped-double-Q) estimate
q_tot[k] = sum_i Qmin[k,i]                         # team score of joint candidate k

soft:  k* ~ Categorical(logits = q_tot / 0.05)     # Boltzmann over the 8 candidates   <-- what ran
hard:  k* = argmax_k q_tot                         # (soft: False)
execute joint[k*];  mask = choices[k*]             # per-agent 0=IL, 1=RL  -> rl_action_fraction
```

Three properties to state precisely in the paper:

1. **Per-agent mixing.** The arbiter does not pick "the IL policy" or "the RL policy" for the team; it picks
   one of the 2^N per-agent assignments, so a single timestep can execute IL for agent 1 and RL for agents
   2–3. This is the multi-agent generalisation over IBRL's binary switch, and it is what `strict: False`
   means. `strict: True` (`_best_act_strict`) is the ablation that restricts the candidate set to the two
   *all-IL* and *all-RL* joint actions — i.e. plain IBRL lifted to the team.
2. **Boltzmann, not greedy.** With `soft: True, temperature = 0.05` the executed candidate is *sampled* from
   `softmax(q_tot / 0.05)`. Because `q_tot` is a sum of 3 per-agent Q values that live on a scale of tens
   under this reward, a temperature of 0.05 makes the distribution very sharp — near-greedy whenever the
   candidates are well separated, and near-uniform when they are not. This is a second, value-aware source of
   exploration on top of the Gaussian action noise.
3. **Target critics, not online critics.** Selection uses the *target* ensemble at both acting and bootstrap
   time.

Diagnostics computed here and logged: `mean_action_diff = mean|a_RL − a_IL|`, and `mean_q_diff` /
`var_q_diff` = mean and variance over the batch of `q_tot(all-RL) − q_tot(all-IL)` — a direct measure of how
much the critic prefers the RL proposal to the imitation proposal.

### 5.4 Losses (`ibmarl/losses.py: GroupTrainer`)

**Critic (TD(0) with an arbitrated bootstrap).** The next action is not `pi_target(s')`; it is the arbiter's
choice at `s'` (`_best_next_act_comb`), which rebuilds the same 2^N candidate set from the frozen IL policy
and the **target** RL actor, scores it with 2 randomly sampled target critics (min), Boltzmann-samples a
candidate, and returns the corresponding **per-agent** Q values:

```
a'*, q*(s')  = arbiter.bootstrap_proposal(s')            # no grad
y            = r + gamma * (1 - terminated) * q*(s')     # per agent
L_Q          = MSE( Q^(0)(s, a), y )
```

Two things differ from the MADDPG baseline here, deliberately or otherwise:

* the bootstrap maximises over `{IL, RL}^N` instead of following the target actor — this is the mechanism by
  which the demonstration prior raises the value target in states the RL actor has not yet solved;
* **only ensemble member 0 is trained** (`_build_optimisers` builds one Adam over `critics[group][0]`, the
  loss uses `critics[group][0]`, and only member 0 gets a polyak update). Members 1 and 2 stay at
  initialisation for the whole run. See §6.2 — this is the single most important caveat in this document.

**Actor.**

```
L_pi = - mean_over_batch ( sum_over_agents Q^(0)(s, pi_RL(s)) )
```

i.e. exactly the MADDPG actor objective except that it **sums over agents and then averages over the batch**,
whereas TorchRL's `DDPGLoss` averages over both. With N = 3 this makes the IBMARL actor gradient **3× larger**
than the baselines' at the same learning rate. Worth mentioning if actor step-size is ever discussed as a
confound.

**Updates.** Per optimiser step: critic step (`zero_grad → backward → clip 1.0 → step`), then actor step, then
`polyak_step` on critic member 0 and on the actor with τ = 0.001. `polyak_step` already has a `ModuleList`
branch that would update the whole ensemble; it is simply never given the ensemble at `e0da62b`.

### 5.5 Replay — single buffer, IBRL-style (`ibmarl/data.py`)

Unlike RLfD there is **no demo/online split and no annealing schedule**. One buffer per group of capacity
`ibmarl_replay_size = 200 000` (or `memory_size = 1e6` when the key is absent, as in the archived run
configs), pre-loaded with the 2400 demo transitions; online transitions are appended and the demo share
decays naturally as `2400 / (2400 + 1000·t)` — 71 % at iteration 1, 19 % at iteration 10, 2 % at iteration
100. At 200 000 capacity the ring buffer does not begin evicting until iteration ≈ 198 of 250; at 1e6 it
never evicts. Online batches are `select`ed down to the demo buffer's key schema before insertion so the two
sources share one storage layout.

**Warm-up.** Training and evaluation are skipped while `len(buffer) < ibmarl_min_warm_up_frames = 5000`.
With 2400 demos pre-loaded and 1000 frames added per iteration, the buffer reaches 3400 / 4400 / 5400 after
iterations 0 / 1 / 2 — so **iterations 0 and 1 are pure collection** (blank actor/critic loss and no eval
rows) and gradient updates begin at iteration 2. During those two iterations the arbiter is already active
and scoring with untrained critics.

### 5.6 Ablation switches already wired up

| Config key | Effect |
| --- | --- |
| `strict: True` | candidate set reduced to all-IL vs all-RL (IBRL-style binary switch) |
| `soft: False` | greedy `argmax` over candidates instead of Boltzmann sampling |
| `temperature` | Boltzmann sharpness (0.05 as run) |
| `num_critics: 1` | ensemble off; arbiter scores with the single trained critic (see §6.2) |
| `exploration_noise.sigma_*`, `--sigma_init/--sigma_end` | Gaussian action noise, applied to **both** IL and RL candidates |

---

## 6. Implementation/algorithm mismatches at `e0da62b`

These are properties of the code that produced the results. They are stated here so the paper describes what
was actually run — each was later identified and fixed in `42dc2e5`, which is *excluded* from this report by
assumption.

### 6.1 Demonstration transitions are self-transitions (affects **IBMARL** and **RLfD**; not RFT)

The demo loader builds the `"next"` subtree as `agents_data.clone()`, so
`next_obs = obs` for every demonstration transition, even though the `.pt` file contains a real `next_obs`
array (and an all-False `dones`). The TD target on a demo sample therefore reduces to

```
Q(s,a)  <-  r + gamma * Q(s, a'(s))
```

whose fixed point is `r / (1 - gamma) = 100·r`. With the sparse reward this teaches the critic that
demonstration states are absorbing: −100 for every out-of-ball demo state, and a large positive value for
in-ball ones. For **IBMARL** the same distorted critic is what the arbiter uses to choose between IL and RL
proposals, so the effect propagates into action selection, not only into value estimates. For **RLfD** it
corrupts 50 %→0 % of each minibatch. **RFT is unaffected**, because its demo buffer only ever feeds the BC
MSE, which reads `obs` and `act` and never touches the `next` subtree — a useful asymmetry when comparing
the three demo-based methods.

### 6.2 Only 1 of the 3 critics is trained, but the arbiter samples 2 of 3

`num_critics = 3` builds three critics and three targets, but only member 0 has an optimiser, a loss and a
polyak update. Members 1 and 2 remain at random initialisation for the entire run, as do their targets. The
arbiter samples **2 distinct indices out of 3** on every call, so:

* with probability 2/3 the sampled pair is `{0, x}` — the pessimistic min is taken between the trained critic
  and an untrained one;
* with probability **1/3** the pair is `{1, 2}` — **both members are untrained**, and both the executed action
  and (at the corresponding bootstrap call) the TD target are decided by random networks.

Direction of the effect, stated as a hypothesis the paper can qualify: an untrained Tanh-MLP critic outputs
values near 0, while a trained critic under this reward outputs large negative values, so the `min` will
usually return the *trained* member's estimate whenever member 0 is in the pair. The damage is concentrated
in the 1/3 of calls that draw `{1, 2}`, where `q_tot` is ≈ noise — making the Boltzmann selection
approximately uniform over the 8 candidates (so `rl_action_fraction` is pulled toward 0.5) and injecting
targets of roughly `y ≈ r` into the critic regression. This is consistent with the observed
`rl_action_fraction ≈ 0.49` at iteration 0.

**Implication for the write-up:** the ensemble as run is *not* a working clipped-double-Q / pessimistic
ensemble. Either describe the method as using a single trained critic with a stochastic corruption in the
arbiter, or re-run with `num_critics: 1` (which is a supported config and yields a clean single-critic
arbiter identical in spirit to IBRL) before claiming an ensemble contribution.

### 6.3 Exploration-noise scheduler stepped inside the inner loop (IBMARL only)

`noise_modules[group].step(current_frames)` and the IL-noise equivalent are called inside the 500-step
optimiser loop rather than once per iteration, advancing the annealing schedule by
`500 × 1000 = 500 000` frames per iteration against an `annealing_num_steps` of `total_frames // 2 =
125 000`. **This is inert for the reported runs** because `sigma_init == sigma_end == 0.1`, but any run using
`--sigma_init/--sigma_end` would have σ saturated at `sigma_end` within iteration 0. Check whether any
sweep in the paper used those flags.

### 6.4 IL noise is applied when acting but not when bootstrapping

`_best_act_comb` adds Gaussian noise to the IL candidate; `_best_next_act_comb` does not (it also uses the
target actor rather than the online actor for the RL candidate, which is intended). Minor, but it means the
acting and bootstrap candidate distributions are not identical.

### 6.5 `eval_reward_mean` / `rl_only_episode_reward_mean`

See §2.7. In short: the deterministic-evaluation column produced by this revision is mis-computed (≈ half the
true return) and `rl_only_episode_reward_mean` is a copy of it, not an RL-only measurement. **The result
CSVs in this repo do not contain these columns**, so the safe description of your curves is
"mean per-agent episode return of the behaviour policy over episodes completed in each 1000-frame collection
batch" — noting that for IBMARL the behaviour policy is the arbitrated IL/RL mixture, so its curve is *not*
directly comparable to an RL-policy-only curve.

### 6.6 `analysis/training_scripts/gt_rewards.bash` was non-functional

It passes `--seed`, which `run_experiment.py`'s argparse rejects as ambiguous against `--seeds` /
`--seed-start`; all runs launched through it exited with code 2 before training. Relevant only if the paper
claims those sweeps were run from that script.

---

## 7. One-line summary of what distinguishes each method

| Method | Demonstrations used for | Changed relative to MADDPG |
| --- | --- | --- |
| **MADDPG** | — | (reference) |
| **RLfD** | replay sampling | minibatch = demo fraction annealed 50 %→0 % over training; losses identical |
| **RFT** | actor regularisation | `L_actor += 0.1 · max(0, 1 − t/150) · MSE(pi(s_demo), a_demo)`; buffers identical |
| **IBMARL** | frozen policy **and** replay seeding | frozen BC policy proposes actions; a critic-scored arbiter Boltzmann-samples over the 2^N per-agent IL/RL joint combinations, at both acting time and in the TD bootstrap; single replay buffer pre-loaded with demos; 3-member critic ensemble (see §6.2); actor objective sums over agents |

## 8. Reproduction

```bash
python -m src.run_experiment navigation maddpg   --seeds 5 --seed-start 0
python -m src.run_experiment navigation rlfd     --seeds 5 --seed-start 0
python -m src.run_experiment navigation rft      --seeds 5 --seed-start 0
python -m src.run_experiment navigation ibmarl   --seeds 5 --seed-start 0
python -m src.run_experiment navigation bc_eval                     # imitation-prior reference
```

Each writes `results/<exp>_<scenario>_<timestamp>/` containing `config.yaml` (the fully merged config
actually used), `data/metrics.csv`, `plots/episode_rewards.png`, `checkpoints/policy_checkpoint.pt` (actor
state dicts; IBMARL saves the **RL** policies only), and `videos/`.

## 9. Open items the paper author should resolve

1. Which runs back each figure, and whether they came from this revision or the pre-force-push `main`
   (algorithmically identical, §0).
2. Whether the ensemble is claimed as a contribution — §6.2 says it cannot be, as run.
3. Whether any reported curve uses `eval_reward_mean` (§2.7); if so its scale is wrong.
4. How the IBMARL curve is labelled: it is the arbitrated mixture's return, not the RL policy's.
5. Whether `--sigma_init/--sigma_end` were used in any sweep (§6.3).
6. Demo budget to quote: **24 demonstrations, 2400 transitions**, 3 agents, from the R2BC decentralised
   BC policy (per-agent 18→8→2 MLP). The `config/experiments/*.yaml` at this revision point at a
   collaborator's home directory (`/home/connor/...`) for `ibmarl`, and at
   `/home/tony-zhang/.../navigation_r2bc_decent_20260129_201300/` for `rft`/`rlfd`; the archived run configs
   under `results/` point at `navigation_r2bc_decent_20260130_122306`. All three demo files carry the same
   2400×3×18 shape and near-identical statistics, but **confirm which checkpoint produced the reported
   numbers** — the IBMARL and baseline runs must share one to be a controlled comparison.
