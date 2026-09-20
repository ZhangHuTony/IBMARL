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
| buzz_wire | `buzzwire_r2bc_decent_20260826_225653` — 12 demos (`--total_demonstrations 12 --total_samples 6`) | same four |
| balance | `balance_r2bc_decent_20260823_185058` | `policy_checkpoint.pth`, `demonstrations.pt`, `metadata.json` |
| transport | `transport_r2bc_decent_20260821_154402` | `policy_checkpoint.pth`, `metadata.json` — **demos not bundled** |

`metadata.json` is R2BC's own record of the collection command line.  The runs'
`config.yaml`, `metrics.csv` and media are deliberately left behind: they carry
nothing the loaders read and are full of dead collaborator paths.

## Reward schema of the recordings

Every `demonstrations.pt` was recorded under the **legacy sparse schema**:
`-1` per step while off-goal, the native VMAS reward inside the basin, fixed
horizon with `done()` suppressed, so `dones` is all-False and episodes are
consecutive `horizon`-length blocks per sub-env (rows are time-major /
env-minor: `index = t * n_envs + e`).

Navigation and buzz_wire now use the **binary terminal schema**
(`binary_terminal_reward: True` in `config/environments/`): +1 once, on the
first step the success predicate holds, termination there, 0 otherwise.
`demonstrations_binary.pt` is derived from the legacy recording by

    python -m analysis.relabel_demos_binary navigation
    python -m analysis.relabel_demos_binary buzz_wire

without re-simulating (navigation recovers per-agent goal distance from the
observation; buzz_wire recovers the basin from `reward != -1`), and carries an
extra `terminated` array plus a `meta` dict describing the result.  Balance
still uses the legacy schema and loads `demonstrations.pt` directly.

Caveat for buzz_wire: online, wall contact ends the episode with 0, but the
recording cannot reveal collisions (the ball is not in the observation), so
relabelled episodes are cut at the basin regardless of an earlier touch.  The
teacher collides before success in roughly 8% of its successful episodes, so
the demo file is optimistic by about that much.  Kept as is rather than
re-collecting, which would also have changed the teacher (2026-09-19).

## Transport demonstrations

`transport_r2bc_decent_20260821_154402/demonstrations.pt` is 91 MB — over
GitHub's 50 MB warning and near its 100 MB hard limit, and this repository does
not use Git LFS.  The transport experiment configs therefore keep an absolute
`demonstrations_path`; on another machine, copy that file from the R2BC run
and point `config/experiments/{ibmarl,rlfd,rft}_transport.yaml` at it.
`ibmarl` warm-up/arbitration and `bc_eval` need only the bundled checkpoint.
