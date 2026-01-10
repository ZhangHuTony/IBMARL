"""
Main script to run experiments for the IBMARL project.
"""
import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

from src.experiment_registry import EXPERIMENT_REGISTRY


def load_config(scenario_name: str, exp_type: str) -> dict:
    """
    Loads and merge yaml configuration files based on scenario name and experiment type.
    """


    # Load base config
    base_config_path = Path("config/base.yaml")
    print(f"Loading base config from: {base_config_path.absolute()}")
    with open(base_config_path) as f:
        config = yaml.safe_load(f)
    print("Base config training params:", config.get('training', {}))
    
    # Load environment-specific config
    env_config_path = Path(f"config/environments/{scenario_name}.yaml")
    if env_config_path.exists():
        with open(env_config_path) as f:
            env_config = yaml.safe_load(f)
            
            # Properly merge training parameters
            if 'training' in env_config:
                if 'training' not in config:
                    config['training'] = {}
                config['training'].update(env_config.pop('training'))
            
            # Update rest of config
            config.update(env_config)
    
    print("Final merged training params:", config.get('training', {}))
    config['total_frames'] = config.get('frames_per_batch', 1000) * config.get('n_iters', 10)

    config['scenario_name'] = scenario_name
    config['exp_type'] = exp_type
    print(config)
    return config

def run_experiment(cfg):
    """
    Run the IBMARL experiment based on the provided arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        results: The results of the experiment.
    """
    # Placeholder for experiment logic
    print(f"Running experiment {cfg['exp_type']} for scenario: {cfg['scenario_name']}")

    ExperimentClass = EXPERIMENT_REGISTRY.get(cfg['exp_type'])
    if ExperimentClass is None:
        raise ValueError(f"Experiment type {cfg['exp_type']} not found in registry.")

    experiment = ExperimentClass(cfg)
    results = experiment.train()

    if cfg.get("render"):
        experiment.render()
    return results

def plot(cfg, rewards, group_map):
    fig, axs = plt.subplots(len(group_map), 1, figsize=(6, 4 * len(group_map)))

    # Make axs iterable when there is only one group
    if len(group_map) == 1:
        axs = [axs]

    for i, group in enumerate(group_map):
        axs[i].plot(rewards[group], label=f"Episode reward mean {group}")
        axs[i].set_ylabel("Reward")
        axs[i].legend()

    axs[-1].set_xlabel("Training iterations")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run IBMARL Experiments")

    #Required arguments
    parser.add_argument("scenario_name", type=str, help="Name of the VMAS scenario to run")
    parser.add_argument("exp_type", type=str, help= "ID for experiment")


    #Optional Arguments (can override YAML)
    parser.add_argument("--render", action="store_true", help="Enable rendering")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.scenario_name, args.exp_type)

    cfg["render"] = bool(args.render)
    # Run the experiment
    rewards, group_map = run_experiment(cfg)

    plot(cfg, rewards, group_map)

    