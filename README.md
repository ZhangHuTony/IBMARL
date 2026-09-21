Using https://docs.pytorch.org/rl/0.4/tutorials/multiagent_competitive_ddpg.html as a reference


To run maddpg:
 `python -m src.run_experiment navigation maddpg --render`

To run ibmarl:
 `python -m src.run_experiment navigation ibmarl --render`

To run RLfD:
 `python -m src.run_experiment navigation rlfd --render`

To change parameters of experiment look into config yaml files.



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