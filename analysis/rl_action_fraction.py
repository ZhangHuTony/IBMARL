"""Plot rl_action_fraction over iterations for IBMARL experiments, averaged across seeds."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG: dict[str, dict[str, str]] = {
    "navigation_dense": {
        "ibmarl": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/ibmarl",
        # "ibmarl (noisy IL)": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison/ibmarl_il_explore",
    },
    "navigation_sparse": {
        "ibmarl": "/home/connor/Desktop/Projects/IBMARL/results/rlfd_rl_comparison_sparse/ibmarl",
    },
}


def discover_seed_dirs(parent_dir: str) -> list[Path]:
    parent = Path(parent_dir) if Path(parent_dir).is_absolute() else PROJECT_ROOT / parent_dir
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory not found: {parent}")
    seed_dirs = [
        p for p in sorted(parent.iterdir())
        if p.is_dir() and (p / "data" / "metrics.csv").exists()
    ]
    if not seed_dirs:
        raise FileNotFoundError(f"No seed dirs with data/metrics.csv found in {parent}")
    return seed_dirs


def load_metric(parent_dir: str, metric: str = "rl_action_fraction") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_dirs = discover_seed_dirs(parent_dir)
    dfs = []
    for exp_dir in seed_dirs:
        path = exp_dir / "data" / "metrics.csv"
        df = pd.read_csv(path)
        dfs.append(df[["iteration", metric]].rename(columns={metric: "value"}))

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
    values = merged[run_cols].values
    n_runs = np.maximum(np.sum(~np.isnan(values), axis=1), 1)
    mean = np.nanmean(values, axis=1)
    stderr = np.nanstd(values, axis=1, ddof=1) / np.sqrt(n_runs)
    stderr = np.where(np.isnan(stderr), 0, stderr)

    return iterations, mean, stderr


def main() -> None:
    n_envs = len(CONFIG)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 6), squeeze=False)
    axes = axes.flatten()

    for idx, (env_name, experiments) in enumerate(CONFIG.items()):
        ax = axes[idx]
        for label, parent_dir in experiments.items():
            try:
                iterations, mean, stderr = load_metric(parent_dir)
                print(f"Plotting {label} for {env_name}: {len(mean)} datapoints, {len(discover_seed_dirs(parent_dir))} seeds")
                ax.plot(iterations, mean, label=label)
                ax.fill_between(iterations, mean - stderr, mean + stderr, alpha=0.3)
            except FileNotFoundError as e:
                print(f"Skipping {label} for {env_name}: {e}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("RL Action Fraction")
        ax.set_title(env_name)
        ax.legend()
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
