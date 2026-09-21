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
| balance | `balance_r2bc_decent_20260823_185058` | `policy_checkpoint.pth`, `demonstrations.pt`, `metadata.json` |
| transport | `transport_r2bc_decent_20260821_154402` — 204 episodes of the suboptimal heuristic, dense rewards | `policy_checkpoint.pth`, `demonstrations.pt` (float32 tensors, 36.8 MB; bundled 2026-09-21), `metadata.json` |

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
relabelled episodes are cut at the basin regardless of an earlier touch, and
the demo file is optimistic by however often that happens.  For the current
teacher this matters less than it did: it is weak by being SLOW (demonstrator
gain 0.9), so its failures are mostly arrivals past the 200-step horizon --
3 of its 6 demos never reach the basin at all, and the other three arrive at
steps 42 / 137 / 173.  Contact rate for this teacher is unmeasured.

Teacher history on this task (6 demos throughout since 2026-09-20; measured
under the binary schema, 400 fresh episodes):

* 2026-09-20: 12-demo -> 6-demo teacher, tuned demonstrator (gain 4.0):
  0.882 -> 0.770.  Demo count is a cliff (2: 0.010, 4: 0.037, 6: 0.770), so it
  could not go lower.
* 2026-09-21: demonstrator gain 4.0 -> 0.9, same 6-demo budget: 0.765 -> 0.492
  +/- 0.025.  The gain dial is non-monotone -- 2.5 gives the STRONGEST teacher
  (0.948, fewer wire contacts) and only below ~1.5 does slowness dominate; the
  full grid is in `config/experiments/ibmarl_buzz_wire.yaml`.  The
  demonstrations are swapped with the checkpoint (same run), so the demo set
  the RLfD/RFT baselines and the IBMARL buffer pre-load consume changed too:
  952 relabelled transitions, 3 of 6 episodes reaching the goal (buzzwire4:
  855 and 4 of 6).  Curves recorded before this date used the gain-4.0 teacher.

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
