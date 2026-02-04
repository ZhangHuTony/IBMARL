"""Plot main results: mean reward across seeds with standard error regions."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project root (parent of analysis/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config: env/task name -> {method name -> list of experiment directories}
# Experiment dirs are relative to PROJECT_ROOT; each contains data/metrics.csv
CONFIG: dict[str, dict[str, list[str]]] = {
    "navigation": {
        "maddpg": [
            "results/initial_tests/maddpg_navigation_s1",
            "results/initial_tests/maddpg_navigation_s2",
            "results/initial_tests/maddpg_navigation_s3",
            "results/initial_tests/maddpg_navigation_s4",
            "results/initial_tests/maddpg_navigation_s5",

        ],
        "ibmarl": [
            # "results/maddpg_navigation_2026-01-29_13-32-46",
            # Add more seed runs here
        ],
    },
    # Add more environments as needed:
    "balance": {
        "maddpg": [
            "results/initial_tests/maddpg_balance_s1",
            "results/initial_tests/maddpg_balance_s2",
            "results/initial_tests/maddpg_balance_s3",
            "results/initial_tests/maddpg_balance_s4",
            "results/initial_tests/maddpg_balance_s5",
        ],
    },
    # "other_env": {
    #     "maddpg": ["results/s42_other_maddpg_gt_reward"],
    #     "ibmarl": ["results/s42_other_ibmarl_gt_reward"],
    # },
}


def load_metrics_for_method(exp_dirs: list[str], metric: str = "episode_reward_mean") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load metrics from multiple experiment directories (seeds) and compute
    mean and standard error across seeds.

    Returns:
        iterations: 1D array of iteration values
        mean: mean across seeds at each iteration
        stderr: standard error (std / sqrt(n)) at each iteration
    """
    dfs = []
    for exp_dir in exp_dirs:
        path = PROJECT_ROOT / exp_dir / "data" / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
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
        for method_name, exp_dirs in methods.items():
            if not exp_dirs:
                continue
            try:
                iterations, mean, stderr = load_metrics_for_method(exp_dirs)
                print("Plotting", method_name, "for", env_name, "with", len(mean), "datapoints")
                ax.plot(iterations, mean, label=method_name)
                ax.fill_between(
                    iterations,
                    mean - stderr,
                    mean + stderr,
                    alpha=0.3,
                )
            except FileNotFoundError as e:
                print(f"Skipping {method_name} for {env_name}: {e}")

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
