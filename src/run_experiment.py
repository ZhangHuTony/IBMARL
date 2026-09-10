"""
Main script to run experiments for the IBMARL project.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from src.util import PushoverNotifier

from src.experiment_registry import EXPERIMENT_REGISTRY


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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

    #Load experiment-specific config
    exp_config_path = Path(f"config/experiments/{exp_type}.yaml")
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_config = yaml.safe_load(f)

            config.update(exp_config)

    # Per-scenario experiment overlay, e.g. config/experiments/ibmarl_transport.yaml.
    # The experiment yamls hold absolute teacher-artifact paths that are only valid for
    # one scenario; this layer lets a second scenario supply its own without touching
    # the navigation defaults.  Unlike the layer above, `training` is merged key-by-key
    # (as the environment layer does) so an overlay cannot clobber the whole block.
    exp_scenario_path = Path(f"config/experiments/{exp_type}_{scenario_name}.yaml")
    if exp_scenario_path.exists():
        print(f"Loading per-scenario experiment config from: {exp_scenario_path}")
        with open(exp_scenario_path) as f:
            exp_scenario_config = yaml.safe_load(f) or {}

        if "training" in exp_scenario_config:
            config.setdefault("training", {}).update(exp_scenario_config.pop("training"))

        config.update(exp_scenario_config)

    config['scenario_name'] = scenario_name
    config['exp_type'] = exp_type
    config['total_frames'] = config.get('frames_per_batch', 1000) * config.get('n_iters', 10)
    print(config)
    return config

def create_run_dirs(cfg: dict):
    """
    creates unique run directory under results/ to save experiment data
    """

    # Annealing horizons describe the original experiment and must not move if
    # a completed run is later extended to a larger n_iters target.
    cfg.setdefault(
        "exploration_annealing_frames", int(cfg.get("total_frames", 0)) // 2
    )
    if cfg.get("exp_type") == "rlfd":
        cfg.setdefault("demo_anneal_n_iters", int(cfg["n_iters"]))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg['exp_type']}_{cfg['scenario_name']}_{timestamp}"

    results_root = Path("results")
    run_dir = results_root / run_name

    data_dir = run_dir / "data"
    check_dir = run_dir / "checkpoints"
    videos_dir = run_dir / "videos"
    plots_dir = run_dir / "plots"
    artifacts_dir = run_dir / "artifacts"


    # modify so that it makes the directories after training
    for d in (data_dir, check_dir, videos_dir, plots_dir, artifacts_dir):
       d.mkdir(parents=True, exist_ok=False)

    
    cfg["run_name"] = run_name
    cfg["run_dir"] = str(run_dir)
    cfg ["data_dir"] = str(data_dir)
    cfg["check_dir"] = str(check_dir)
    cfg ["videos_dir"] = str(videos_dir)
    cfg["plots_dir"] = str(plots_dir)

    # Demonstration-based experiments otherwise depend on files elsewhere in
    # results/ (or even another checkout).  Copy them into the run so a future
    # resume only needs this directory.
    artifact_manifest = {}
    for key in ("r2bc_checkpoint_path", "demonstrations_path"):
        source_value = cfg.get(key)
        if not source_value:
            continue
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Configured input artifact does not exist: {source}")
        destination = artifacts_dir / f"{key}{source.suffix}"
        shutil.copy2(source, destination)
        artifact_manifest[key] = {
            "original_path": str(source),
            "saved_path": str(destination.resolve()),
            "sha256": _sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
        cfg[key] = str(destination.resolve())

    if artifact_manifest:
        with open(run_dir / "artifacts.yaml", "w") as f:
            yaml.safe_dump(artifact_manifest, f, sort_keys=False)

    # Save the exact merged config used for this run
    config_out = run_dir / "config.yaml"
    with open(config_out, "w") as f:
       yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Run directory created: {run_dir.resolve()}")
    return cfg


def load_resume_config(run_dir_value: str, n_iters: int | None = None) -> dict:
    """Load one existing run and point all output paths back into it."""
    run_dir = Path(run_dir_value).expanduser().resolve()
    config_path = run_dir / "config.yaml"
    state_path = run_dir / "checkpoints" / "resume" / "state.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"No run config found at {config_path}")
    if not state_path.is_file():
        model_files = sorted((run_dir / "checkpoints").glob("*_checkpoint.pt"))
        detail = (
            " Model-only files found: " + ", ".join(p.name for p in model_files)
            if model_files
            else ""
        )
        raise FileNotFoundError(
            f"No full training state found at {state_path}; this run cannot be "
            f"resumed exactly.{detail}"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    state = torch.load(state_path, map_location="cpu", weights_only=False)

    # Rebase copied inputs if the whole run directory has moved, and detect
    # corruption before constructing policies or replay buffers from them.
    manifest_path = run_dir / "artifacts.yaml"
    if manifest_path.is_file():
        with open(manifest_path) as f:
            artifact_manifest = yaml.safe_load(f) or {}
        for key, entry in artifact_manifest.items():
            artifact = run_dir / "artifacts" / Path(entry["saved_path"]).name
            if not artifact.is_file():
                raise FileNotFoundError(f"Saved run artifact is missing: {artifact}")
            actual_digest = _sha256_file(artifact)
            if actual_digest != entry["sha256"]:
                raise ValueError(f"Saved run artifact failed checksum: {artifact}")
            cfg[key] = str(artifact.resolve())

    # The CLI only permits changing the target length and rendering behavior.
    # Also guard against an edited config.yaml by comparing it with the config
    # embedded in the checkpoint itself.
    checkpoint_cfg = state.get("config")
    if checkpoint_cfg:
        mutable_keys = {
            "n_iters",
            "total_frames",
            "render",
            "run_name",
            "run_dir",
            "data_dir",
            "check_dir",
            "videos_dir",
            "plots_dir",
            "resume_history",
            "r2bc_checkpoint_path",
            "demonstrations_path",
        }
        saved_immutable = {
            key: value
            for key, value in checkpoint_cfg.items()
            if key not in mutable_keys
        }
        current_immutable = {
            key: value for key, value in cfg.items() if key not in mutable_keys
        }
        if current_immutable != saved_immutable:
            raise ValueError(
                "config.yaml contains training changes incompatible with the full "
                "checkpoint; restore the checkpoint's configuration and change only "
                "--n-iters."
            )

    next_iteration = int(state["iteration"]) + 1
    target_iters = int(n_iters if n_iters is not None else cfg["n_iters"])
    if target_iters <= next_iteration:
        raise ValueError(
            f"Checkpoint has completed {next_iteration} iterations; --n-iters must "
            f"be greater than {next_iteration} to extend this run."
        )

    previous_target = int(cfg["n_iters"])
    cfg["n_iters"] = target_iters
    cfg["total_frames"] = int(cfg["frames_per_batch"]) * target_iters
    cfg["run_dir"] = str(run_dir)
    cfg["data_dir"] = str(run_dir / "data")
    cfg["check_dir"] = str(run_dir / "checkpoints")
    cfg["videos_dir"] = str(run_dir / "videos")
    cfg["plots_dir"] = str(run_dir / "plots")
    cfg.setdefault("resume_history", []).append(
        {
            "resumed_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_iteration": next_iteration - 1,
            "previous_target_iters": previous_target,
            "new_target_iters": target_iters,
        }
    )

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
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
    results_str = experiment.train()

    experiment.save_results()
    if cfg.get("render"):
        experiment.render_policy()

    print(results_str)
    notifier = PushoverNotifier()
    notifier.send_message(results_str)


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
    parser.add_argument("scenario_name", nargs="?", type=str, help="Name of the VMAS scenario to run")
    parser.add_argument("exp_type", nargs="?", type=str, help="ID for experiment (e.g. 'ibmarl', 'maddpg')")


    #Optional Arguments (can override YAML)
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeded runs to execute")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed value (increments by 1 per run)")
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="Continue a run in the same directory from its full training checkpoint",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        help="Total iteration target (required to extend a completed run)",
    )
    reward_mode = parser.add_mutually_exclusive_group()
    reward_mode.add_argument(
        "--sparse",
        dest="sparse_rewards",
        action="store_true",
        default=None,
        help="Force the sparse reward transform on (overrides config.sparse_rewards)",
    )
    reward_mode.add_argument(
        "--dense",
        dest="sparse_rewards",
        action="store_false",
        default=None,
        help="Force the environment's native dense reward (overrides config.sparse_rewards)",
    )
    parser.add_argument(
        "--sigma_init",
        type=float,
        help=(
            "Initial Gaussian exploration noise stddev for IBMARL "
            "(overrides config.exploration_noise.sigma_init; only valid for exp_type='ibmarl')"
        ),
    )
    parser.add_argument(
        "--sigma_end",
        type=float,
        help=(
            "Final Gaussian exploration noise stddev for IBMARL "
            "(overrides config.exploration_noise.sigma_end; only valid for exp_type='ibmarl')"
        ),
    )

    args = parser.parse_args()

    if args.resume:
        if args.scenario_name or args.exp_type:
            parser.error("scenario_name and exp_type must be omitted with --resume")
        if args.seeds != 1 or args.seed_start != 0:
            parser.error("--seeds/--seed-start cannot be used with --resume")
        if args.sparse_rewards is not None or args.sigma_init is not None or args.sigma_end is not None:
            parser.error("training hyperparameters cannot be changed with --resume")
        try:
            cfg = load_resume_config(args.resume, args.n_iters)
        except (FileNotFoundError, ValueError, KeyError) as error:
            parser.error(str(error))
        cfg["render"] = bool(args.render or cfg.get("render", False))
        run_experiment(cfg)
        sys.exit(0)

    if not args.scenario_name or not args.exp_type:
        parser.error("scenario_name and exp_type are required unless --resume is used")

    # Load configuration
    cfg = load_config(args.scenario_name, args.exp_type)

    cfg["render"] = bool(args.render)
    if args.n_iters is not None:
        if args.n_iters <= 0:
            parser.error("--n-iters must be positive")
        cfg["n_iters"] = args.n_iters
        cfg["total_frames"] = int(cfg["frames_per_batch"]) * args.n_iters

    # Reward-mode override.  Unlike --sigma_init/--sigma_end this is not gated
    # on exp_type: every algorithm reads sparse_rewards through make_env.
    if args.sparse_rewards is not None:
        cfg["sparse_rewards"] = bool(args.sparse_rewards)

    # CLI overrides for IBMARL exploration noise
    if args.sigma_init is not None or args.sigma_end is not None:
        if cfg.get("exp_type") != "ibmarl":
            raise ValueError(
                "Command-line arguments --sigma_init/--sigma_end are only valid when exp_type='ibmarl'. "
                f"Got exp_type='{cfg.get('exp_type')}'."
            )
        if "exploration_noise" not in cfg or cfg["exploration_noise"] is None:
            cfg["exploration_noise"] = {}
        if args.sigma_init is not None:
            cfg["exploration_noise"]["sigma_init"] = float(args.sigma_init)
        if args.sigma_end is not None:
            cfg["exploration_noise"]["sigma_end"] = float(args.sigma_end)

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        run_cfg = cfg.copy()
        run_cfg["seed"] = seed
        run_cfg = create_run_dirs(run_cfg)
        run_experiment(run_cfg)
