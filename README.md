Using https://docs.pytorch.org/rl/0.4/tutorials/multiagent_competitive_ddpg.html as a reference


To run maddpg:
 `python -m src.run_experiment navigation maddpg --render`

To run ibmarl:
 `python -m src.run_experiment navigation ibmarl --render`

To run RLfD:
 `python -m src.run_experiment navigation rlfd --render`

To change parameters of experiment look into config yaml files.

### Human Xbox R2BC teachers

Train a decentralized R2BC teacher with the paper's round-robin interaction:
one agent is controlled with the Xbox left stick while every other agent runs
its current cloned policy. Press **A** to begin each episode, **B** to reject the
current episode, **START** to save and quit, and hold **RT** for precision mode.
Agents are persistently colored and identified by the in-window legend:
**Agent 1 = blue, Agent 2 = orange, Agent 3 = green**. The currently controlled
agent is also named in the window before each episode.

```bash
# Local sparse binary-terminal buzz wire (2 agents, 200-step horizon)
python -m src.r2bc.human_teleop buzz_wire --total-demonstrations 24

# Default dense-reward transport (3 agents, 500-step horizon)
python -m src.r2bc.human_teleop transport --total-demonstrations 48

# Continue an autosaved run in place, up to 24 total accepted demonstrations
python -m src.r2bc.human_teleop buzz_wire --total-demonstrations 24 \
  --resume results/buzz_wire_r2bc_human_20260923_155838
```

On resume, `--total-demonstrations` is the cumulative target, not the number of
additional demonstrations. The saved environment/configuration, policy,
demonstrations, round-robin position, and (for runs saved by the current code)
optimizer state are restored. Older human-R2BC runs do not have
`training_state.pth`; they can still be resumed, with fresh Adam optimizer
state and all prior demonstrations retained for cumulative BC training.

The run directory is autosaved after every accepted episode and contains both
`policy_checkpoint.pth` (directly loadable by `R2bcPolicy`) and
`demonstrations.pt` with the existing `obs / act / rewards / next_obs / dones`
schema. It additionally stores `terminated` and round-robin agent metadata;
existing IBMARL/RLfD/RFT loaders ignore the metadata and consume the file as-is.
Point the relevant experiment overlay's `r2bc_checkpoint_path` and
`demonstrations_path` at these two files to use the human teacher.

The human buzz-wire command uses
`src/environment/scenarios/buzz_wire_sparse.py`, including its configured
binary terminal success/collision semantics. Its saved `config.yaml`,
`demonstrations.pt`, and `metadata.json` identify the reward mode as sparse.



File Structure:
```
IBMARL/
├──config/                                                          # holds the yaml which holds the parameters used for experiments
├──results/                                                         # holds the results of experiments
├──teachers/                                                        # bundled R2BC teacher checkpoints + demonstrations (see teachers/README.md)
├──src/                                                             # holds all the files needed to run an experiment
|   ├── run_experiment.py                                           # entry point for training
|   ├── experiment_registry.py                                      # retrieves experiment classes 
|   ├── experiments/                                                # Holds the classes for all runnable experiments
|      ├── base_marl_experiment.py                                 # base marl experiment class for all marl experiments
|      ├── maddpg.py                                               # holds all needed logic for running MADDPG
|      ├── ibmarl_experiment.py                                    # driver for IBMARL, uses logic made in ibmarl folder.
|      └── rlfd_experiment.py                                      # RL from Demonstrations experiment (RLfD)
|   ├── environment/                                                # holds files needed to construct the environment used for MARL
|      ├── make_env.py                                             # constructs the environment
|      └── transforms/                                             # holds experiment specific transformations to the environment
└──requirements.txt
```



### Running a sweep on a fresh machine

Everything a run needs is in the checkout: the frozen teachers and their
demonstrations are bundled under `teachers/` (see `teachers/README.md`), and the
config artifact paths are repo-relative, resolved against the checkout by
`src/util/paths.py`.  Transport's demonstrations, once a 95 MB machine-local file,
are bundled too (re-saved as float32 tensors, 36.8 MB), so every task runs from
the checkout alone.

```bash
git clone https://github.com/ZhangHuTony/IBMARL.git && cd IBMARL
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # versions are pinned; see the file header

# always launched from the repository root -- load_config reads config/ relatively
python -m analysis.paper_sweep --scenario buzz_wire --tag <tag> \
    --only <variant,variant,...> --jobs <concurrent runs>
```

`--only` names the variants (`analysis/paper_run.py::VARIANTS`); the bare
`paper_sweep.JOBS` table is the navigation paper sweep and is not the buzz-wire
plan.  Variants named in `--only` that are absent from `JOBS` get `MAIN_SEEDS`
(5 seeds), which is how the `ibmarl_strict_gated_a*` actor-lag variants run.

The sweep is resumable at two levels -- a finished run is skipped by its
`status.json`, an interrupted one restarts from its last
`checkpoints/resume` point -- so re-running the same command after a preemption
continues where it stopped.  `--status` prints progress without launching
anything, and `--dry-run` lists what would run.

Buzz-wire and navigation both run the binary terminal reward schema
(`binary_terminal_reward` in `config/environments/`), so their numbers are
success rates in [0, 1] and are NOT comparable to results recorded before
2026-09-19 under the legacy -1/step schema.

### Push Notifications
To recieve notifications about experiments, create a Pushover account, create an application, and export your keys as environment variables in your bash setup (or whatever shell you use).
```
echo "export MACHINE_NAME=<NAME_YOUR_MACHINE>" >> ~/.bashrc
echo "export PUSHOVER_APP_TOKEN=<YOUR_APP_TOKEN>" >> ~/.bashrc
echo "export PUSHOVER_USER_KEY=<YOUR_USER_KEY>" >> ~/.bashrc 
```
