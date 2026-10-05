# Bundled R2BC teachers

Frozen imitation policies (the **teacher** in CONTEXT.md) and the demonstrations
they were trained from, copied out of R2BC run directories so that this
repository runs on any machine without the R2BC tree.  Nothing here is needed
beyond the two artifact files: `R2bcPolicy` rebuilds the network from the
fields embedded in `policy_checkpoint.pth` (`src/r2bc/mabc.py`, a vendored
copy of the R2BC model code), and `demonstrations.pt` is a self-contained dict
of `obs / act / rewards / next_obs / dones` arrays.

Config keys point here with repo-relative paths (`teachers/<task>/...`),
resolved against the checkout by `src/util/paths.py` regardless of the
working directory.

| task | R2BC run (source of every file below) | files |
|---|---|---|
| navigation | `navigation_r2bc_decent_20260129_201300` — 24 demos, collected against a 0.4-radius sparse basin | `policy_checkpoint.pth`, `demonstrations.pt`, `demonstrations_binary.pt`, `metadata.json` |
| buzz_wire | `buzzwire_r2bc_decent_20260921_021440` — 6 demos of a detuned demonstrator, gain 0.9 (`--total_demonstrations 6 --total_samples 3 --move_factor 0.9`) | same four |
| balance | `balance_r2bc_decent_20260823_185058` — 36 demos (10,800 rows) of a MAPPO-supervised demonstrator, DENSE VMAS reward, fixed 300-step horizon, 4-env interleaved | `policy_checkpoint.pth`, `demonstrations.pt` (raw, dense), `demonstrations_sparse.pt` (gt-in-radius relabel the sweeps load; see below), `metadata.json` |
| transport | `transport_r2bc_decent_20260821_154402` — 204 episodes of the suboptimal heuristic, dense rewards | `policy_checkpoint.pth`, `demonstrations.pt` (float32 tensors, 36.8 MB; bundled 2026-09-21), `metadata.json` |
| buzz_wire_12demo | `buzzwire_r2bc_decent_20260826_225653` — 12 demos of the tuned demonstrator (gain 4.0); the teacher of the legacy-schema sweeps buzzwire3 / buzzwire3_reg (bc_eval -137.5), re-bundled 2026-09-24 for the poster's legacy runs (`config/legacy/buzz_wire/*.yaml`) | `policy_checkpoint.pth`, `demonstrations.pt`, `metadata.json` |
| human_buzz_wire_24_demos | human Xbox teleoperation, 2026-09-23 (`src/r2bc/human_teleop.py` on the collaborator's checkout, branch `feat-human-demos`): 24 round-robin episodes, one agent driven per episode, 13 of 24 reach the basin; legacy -1/step schema, fixed 200-step horizon | `policy_checkpoint.pth` (BC teacher trained at collection, bc_eval -143.4 +/- 0.7 over 3 seeds), `demonstrations.pt` (raw, format_version 2), `demonstrations_legacy.pt` (what the loaders read, see below), `metadata.json`, `config.yaml`, `training_state.pth` |
| human_transport_90_demos | human Xbox teleoperation, 2026-09-23 (same tool): 90 round-robin episodes, 43,828 rows, DENSE reward like transport's live stack, 5 of 90 deliver the package; bundled 2026-10-05 for the `transport_human1` arms (`analysis/paper_run.py::HUMAN_TEACHERS`) | `policy_checkpoint.pth` (BC teacher trained at collection, hidden 32), `demonstrations.pt` (raw, format_version 2, 13.6 MB), `demonstrations_legacy.pt` (prepared, see below; generated on the cluster by `analysis/ensure_demo_artifacts.py` if absent), `metadata.json`, `config.yaml`, `training_state.pth` |

`metadata.json` is R2BC's own record of the collection command line.  The runs'
`config.yaml`, `metrics.csv` and media are deliberately left behind: they carry
nothing the loaders read and are full of dead collaborator paths.

## Reward schema of the recordings

The navigation and buzz_wire `demonstrations.pt` were recorded under the
**legacy sparse schema**: `-1` per step while off-goal, the native VMAS reward
inside the basin, fixed horizon with `done()` suppressed, so `dones` is
all-False and episodes are consecutive `horizon`-length blocks per sub-env
(rows are time-major / env-minor: `index = t * n_envs + e`).  Balance's and
transport's were recorded **dense** (the raw VMAS reward; see their sections
below), at a fixed horizon too.

Navigation and buzz_wire now use the **binary terminal schema**
(`binary_terminal_reward: True` in `config/environments/`): +1 once, on the
first step the success predicate holds, termination there, 0 otherwise.
`demonstrations_binary.pt` is derived from the legacy recording by

    python -m analysis.relabel_demos_binary navigation
    python -m analysis.relabel_demos_binary buzz_wire

without re-simulating (navigation recovers per-agent goal distance from the
observation; buzz_wire recovers the basin from `reward != -1`), and carries an
extra `terminated` array plus a `meta` dict describing the result.

## Balance demonstrations

`teachers/balance/demonstrations.pt` is R2BC's raw recording: 36 episodes of
exactly 300 steps (the recorder suppressed VMAS's fall/on-goal terminal), rows
4-env interleaved, rewards the DENSE VMAS values (100 x the per-step decrease
in package-goal distance, -10 on floor contact; range -10.5 .. 0.86), and
`dones` the recorder's mid-episode on-goal flag (1,684 True rows), not a
terminal.  The task runs the **gt-in-radius sparse schema** since 2026-10
(`config/environments/balance.yaml`: native reward while the package is within
`gt_radius` of the goal, -1 per step otherwise, fixed horizon), so the file the
sweeps load is

    python -m analysis.relabel_demos_sparse balance        # radius = the yaml's gt_radius

-> `demonstrations_sparse.pt`: the same rows with `reward = native if
||next_obs[:, 0, 8:10]|| < gt_radius else -1` (the post-step package-goal
offset, exactly what `BalanceSparseReward` masks on online), `dones` and
`terminated` all-False, the on-goal flag moved to `on_goal`, and a `meta` dict
with the radius and the per-radius statistics.  `gt_radius` is 0.5, chosen a
priori on 2026-10-05 (rationale in the yaml); `analysis/balance_teacher_audit.py`
(task 0 of `slurm/run_balance.slurm`) reports the teacher's basin-reach at
{0.3, 0.5, 0.75, 1.05} for the record.  The file is generated on the cluster by
`analysis/ensure_demo_artifacts.py` (every SLURM task runs it first, and it
regenerates the file if `meta.radius` disagrees with the yaml), so it need not
be committed; `meta.radius` says what a file was built with.  If the teacher
turns out too strong, `analysis/train_bc_subset.py` refits a BC teacher on the
first k episodes into `teachers/balance_<k>demo/`.

## Transport demonstrations

Bundled since 2026-09-21.  R2BC's file for this run is 95 MB on disk because
it pickles per-step lists of arrays; the same 122,400 rows re-saved as float32
tensors are 36.8 MB (values verified identical on round-trip), under GitHub's
50 MB warning, so `config/experiments/{ibmarl,rlfd,rft}_transport.yaml` now use
a repo-relative `demonstrations_path` like every other task.  The recording
keeps two quirks the loader is immune to (see `ibmarl_transport.yaml`): rows
are 2-env interleaved, and `dones` is the package-on-goal flag rather than a
terminal; `load_demonstrations_into_buffer` reads each row as a self-contained
transition and sets `terminated` to all-False.

## Human recordings

`teachers/human_*_demos/demonstrations.pt` are `torch.save` dicts of tensors
(`obs / act / rewards [N,A,1] / next_obs / dones / terminated / meta`), rows
episode-contiguous (not env-interleaved), with `meta.controlled_agent` naming
the human-driven agent of each row.  The recorder flags EVERY episode end as
`terminated`, time limits included, which `src/util/demonstrations.py` would
read as true terminals.  `analysis/prepare_human_demos.py` writes
`demonstrations_legacy.pt` with those flags corrected (buzz_wire: all cleared,
the legacy schema has no terminal; transport: kept only at the 5 delivered
episodes of 90) and everything else identical; `analysis/paper_run.py`'s
`HUMAN_TEACHERS` table points the `*_human` variants at that file per scenario.
The transport recording (`human_transport_90_demos`, 90 episodes, dense
reward, 5 deliveries; 13.6 MB raw + 13.6 MB prepared, under GitHub's 50 MB
warning) is bundled since 2026-10-05 for the `transport_human1` arms
(`slurm/run_balance.slurm`), which run on transport's live dense config stack.
