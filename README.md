Using https://docs.pytorch.org/rl/0.4/tutorials/multiagent_competitive_ddpg.html as a reference


To run maddpg:
 `python -m src.run_experiment navigation maddpg --render`

To run ibmarl:
 `python -m src.run_experiment navigation ibmarl --render`

To run RLfD:
 `python -m src.run_experiment navigation rlfd --render`

To change parameters of experiment look into config yaml files.

### Train an R2BC demonstrator

IBMARL includes the R2BC decentralized demonstrator trainer, so teachers no
longer need to be generated from a separate checkout. Install the optional
R2BC dependencies before loading a learned MAPPO/IPPO/CPPO supervisor. For
MAPPO/IPPO, initialize the two pinned source dependencies as well:

```
python -m pip install -r requirements-r2bc.txt
cd src/r2bc/external_libs
git clone https://github.com/proroklab/HetGPPO.git HetGPPO
git -C HetGPPO checkout 660dc25e46cdcdf217b2d7a192d0b3098940c806
git clone https://github.com/proroklab/rllib_differentiable_comms.git rllib_differentiable_comms
git -C rllib_differentiable_comms checkout f6d4e6085f9d73bbab3c3b785ac75ad011f390a7
cd ../../..
```

These are ordinary local checkouts, not Git submodules. CPPO/`learned`
supervisors only require the Python dependencies; MAPPO/IPPO require both.
MAPPO/IPPO additionally import `wandb` and `torch_geometric` from HetGPPO;
both are pinned in `requirements-r2bc.txt`, so rerun that install after pulling
these changes.
The R2BC requirements pin `setuptools<81` because Ray 2.1 imports its
`pkg_resources` module. If Ray was already installed and reports
`No module named 'pkg_resources'`, rerun
`python -m pip install -r requirements-r2bc.txt` or repair the environment with
`python -m pip install 'setuptools<81'`.

Then train a teacher using an RLlib checkpoint (replace `MAPPO` with the type
used to train the supervisor):

```
python -m src.r2bc.run_experiment balance r2bc --expert_pth PATH_TO_MARL_EXPERT --expert_policy_type MAPPO
```

The bundled runner redirects HetGPPO's hard-coded Linux scratch directory to
`/tmp/ibmarl-ray`, so it does not require a writable `/local` directory or a
local edit to the HetGPPO checkout. To use a different writable location (for
example, a larger scratch disk), set it for the command:

```
IBMARL_RAY_SCRATCH_DIR=/path/to/writable/scratch python -m src.r2bc.run_experiment balance r2bc
```

The run writes `policy_checkpoint.pth` and `demonstrations.pt` under
`results/<scenario>_r2bc_decent_<timestamp>/`. Use the latter as
`demonstrations_path` for IBMARL, RLfD, or RFT. R2BC-specific defaults for all
four supported tasks live in `config/r2bc/`.
They default to `sparse_rewards: true`, so R2BC uses the same sparse reward
definitions from `src/environment/transforms/` (with scenario-level sparse
balance and buzz-wire environments) as IBMARL training. Pass `--dense` only when you
explicitly need native VMAS dense rewards.



File Structure:
```
IBMARL/
├──config/                                                          # holds the yaml which holds the parameters used for experiments
├──results/                                                         # holds the results of experiments
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



### Push Notifications
To recieve notifications about experiments, create a Pushover account, create an application, and export your keys as environment variables in your bash setup (or whatever shell you use).
```
echo "export MACHINE_NAME=<NAME_YOUR_MACHINE>" >> ~/.bashrc
echo "export PUSHOVER_APP_TOKEN=<YOUR_APP_TOKEN>" >> ~/.bashrc
echo "export PUSHOVER_USER_KEY=<YOUR_USER_KEY>" >> ~/.bashrc 
```
