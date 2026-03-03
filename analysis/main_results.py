"""Plot main results: mean reward across seeds with standard error regions."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project root (parent of analysis/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config: env/task name -> {method name -> parent directory or float value}
# Parent dir (string) is relative to PROJECT_ROOT; code scans it for seed subdirs (each with data/metrics.csv)
# Float value draws a dashed horizontal line at that y-value
CONFIG: dict[str, dict[str, str | float]] = {
    
    # DENSE REWARDS
    "navigation_dense": {
        # "maddpg": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/rl",
        # "rlfd": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/rlfd",
        # "rft": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/rft",
        "ibmarl": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/ibmarl",
        "ibmarl (noisy IL)": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/ibmarl_il_explore",
    },
    "navigation_sparse": {
        "maddpg": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison_sparse/rl",
        "rlfd": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison_sparse/rlfd",
        "rft": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison_sparse/rft",
        "ibmarl": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison_sparse/ibmarl",
    },
    # "navigation_dense": {
    #     # "r2bc": -2.18,
    #     "maddpg": "results/maddpg_gt_dense_final",
    #     # "ibmarl (no replay init)": "results/ibmarl_navigation_dense",
    #     # "ibmarl": "results/ibmarl_navigation_dense_w_replay",
    #     # "ibmarl (new results)": "results/temp",
    #     "ibmarl (n=0.05)": "results/ibmarl_005",
    #     "ibmarl (n=0.10)": "results/ibmarl_010",
    #     "ibmarl (n=0.20)": "results/ibmarl_020",
    #     # "ibmarl (n=0.30)": "results/ibmarl_030",
    # },

    # "balance_dense": {
    #     # "r2bc": -2.18,
    #     "maddpg": "results/maddpg_balance_dense",
    #     "ibmarl": "results/ibmarl_balance_dense",
    # },
    # "buzz_wire_dense": {
    #     # "r2bc": -2.18,
    #     "maddpg": "results/maddpg_buzzwire_dense",
    #     # "ibmarl": "results/ibmarl_buzz_wire_dense",
    # },
    # "transport_dense": {
    #     # "r2bc": -2.18,
    #     "maddpg": "results/maddpg_transport_dense",
    #     # "ibmarl": "results/ibmarl_transport_dense",
    # },

    # SPARSE REWARDS
    # "navigation_sparse": {
    #     # "r2bc": -2.18,
    #     "maddpg": "results/maddpg_gt_sparse_final",
    #     "ibmarl": "results/ibmarl_navigation_sparse",
    # },
    # "buzz_wire": {
    #     "maddpg": "results/new_initial_tests/buzz_wire",
    # },
    # "transport": {
    #     "maddpg": "results/new_initial_tests/transport",
    # },
}


def discover_seed_dirs(parent_dir: str) -> list[Path]:
    """Scan parent directory for seed subdirs (direct children that contain data/metrics.csv)."""
    parent = PROJECT_ROOT / parent_dir
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory not found: {parent}")
    seed_dirs = []
    for p in sorted(parent.iterdir()):
        if p.is_dir() and (p / "data" / "metrics.csv").exists():
            seed_dirs.append(p)
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs with data/metrics.csv found in {parent}")
    return seed_dirs


def load_metrics_for_method(parent_dir: str, metric: str = "episode_reward_mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load metrics from all seed subdirs under parent_dir and compute
    mean and standard error across seeds.

    Returns:
        iterations: 1D array of iteration values
        mean: mean across seeds at each iteration
        stderr: standard error (std / sqrt(n)) at each iteration
    """
    seed_dirs = discover_seed_dirs(parent_dir)
    dfs = []
    for exp_dir in seed_dirs:
        path = exp_dir / "data" / "metrics.csv"
        df = pd.read_csv(path)
        dfs.append(df[["iteration", metric]].rename(columns={metric: "value"}))

    # Merge on iteration (handles different lengths via outer join)
    merged = dfs[0][["iteration"]].drop_duplicates().sort_values("iteration")
    for i, df in enumerate(dfs):
        merged = merged.merge(
            df.rename(columns={"value": f"run_{i}"}),
            on="iteration",
            how="outer",
        )

    run_cols = [c for c in merged.columns if c.startswith("run_")]
    merged = merged.sort_values("iteration")

    iterations = merged["iteration"].values
    values = merged[run_cols].values  # shape: (n_iterations, n_runs)
    n_runs = np.sum(~np.isnan(values), axis=1)
    n_runs = np.maximum(n_runs, 1)  # avoid division by zero
    mean = np.nanmean(values, axis=1)
    stderr = np.nanstd(values, axis=1, ddof=1) / np.sqrt(n_runs)
    stderr = np.where(np.isnan(stderr), 0, stderr)

    return iterations, mean, stderr


def main() -> None:
    n_envs = len(CONFIG)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 6), squeeze=False)
    axes = axes.flatten()

    for idx, (env_name, methods) in enumerate(CONFIG.items()):
        ax = axes[idx]
        for method_name, value in methods.items():
            if value is None:
                continue
            if isinstance(value, float):
                # Draw a dashed horizontal line at the float value
                ax.axhline(y=value, linestyle="--", label=method_name, alpha=0.7)
                print(f"Drawing horizontal line for {method_name} at {value} for {env_name}")
            elif isinstance(value, str):
                # Original behavior: load metrics from directory
                if not value:
                    continue
                try:
                    metric = "rl_only_episode_reward_mean" if method_name[:6] == "ibmarl" else "episode_reward_mean"
                    iterations, mean, stderr = load_metrics_for_method(value, metric=metric)
                    print("Plotting", method_name, "for", env_name, "with", len(mean), "datapoints", "using metric", metric)
                    ax.plot(iterations, mean, label=method_name)
                    ax.fill_between(
                        iterations,
                        mean - stderr,
                        mean + stderr,
                        alpha=0.3,
                    )
                except FileNotFoundError as e:
                    print(f"Skipping {method_name} for {env_name}: {e}")
            else:
                print(f"Warning: Unexpected type for {method_name} in {env_name}: {type(value)}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Episode Reward Mean")
        ax.set_title(env_name)
        ax.legend()
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
