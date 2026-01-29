Using https://docs.pytorch.org/rl/0.4/tutorials/multiagent_competitive_ddpg.html as a reference

Trying to get the MADDPG policy to successfully train when running `python -m src.run_experiment navigation maddpg`


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
|      └── ibmarl_experiment.py                                       # driver for ibmarl, uses logic made in ibmarl folder.
|   ├── environment/                                                # holds files needed to construct the environment used for MARL
|      ├── make_env.py                                             # constructs the environment
|      └── transforms/                                             # holds experiment specific transformations to the environment
└──requirements.txt
```


