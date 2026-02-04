"""
Main script to run experiments for the IBMARL project.
"""
import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

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

    #Load experiment-specific config
    exp_config_path = Path(f"config/experiments/{exp_type}.yaml")
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_config = yaml.safe_load(f)

            config.update(exp_config)

    config['scenario_name'] = scenario_name
    config['exp_type'] = exp_type
    print(config)
    return config

def create_run_dirs(cfg: dict):
    """
    creates unique run directory under results/ to save experiment data
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg['exp_type']}_{cfg['scenario_name']}_{timestamp}"

    results_root = Path("results")
    run_dir = results_root / run_name

    data_dir = run_dir / "data"
    check_dir = run_dir / "checkpoints"
    videos_dir = run_dir / "videos"
    plots_dir = run_dir / "plots"


    # modify so that it makes the directories after training
    for d in (data_dir, check_dir, videos_dir, plots_dir):
       d.mkdir(parents=True, exist_ok=False)

    
    cfg["run_name"] = run_name
    cfg["run_dir"] = str(run_dir)
    cfg ["data_dir"] = str(data_dir)
    cfg["check_dir"] = str(check_dir)
    cfg ["videos_dir"] = str(videos_dir)
    cfg["plots_dir"] = str(plots_dir)

    # Save the exact merged config used for this run
    config_out = run_dir / "config.yaml"
    with open(config_out, "w") as f:
       yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Run directory created: {run_dir.resolve()}")
    return cfg



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
    experiment.train()

    experiment.save_results()
    if cfg.get("render"):
        experiment.render_policy()


# WILL REMOVE AND ADD TO AN ANALYSIS SCRIPT
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
    parser.add_argument("--seed", type=int, help="Seed for the experiment")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.scenario_name, args.exp_type)

    cfg["render"] = bool(args.render)
    if args.seed is not None:
        cfg["seed"] = args.seed

    cfg = create_run_dirs(cfg)

    run_experiment(cfg)



    