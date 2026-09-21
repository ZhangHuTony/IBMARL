#!/usr/bin/env python3
"""Plot the rewards stored in an R2BC ``demonstrations.pt`` file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def demonstration_path(experiment: Path) -> Path:
    """Resolve either an R2BC experiment directory or its demonstration file."""
    path = experiment.expanduser()
    if path.is_dir():
        path = path / "demonstrations.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find demonstrations file: {path}")
    return path


def load_rewards(path: Path) -> np.ndarray:
    """Safely load and validate the transition-by-agent reward array."""
    # R2BC saves a dict containing lists of NumPy arrays. These NumPy globals
    # must be explicitly allowlisted for PyTorch's restricted unpickler.
    from numpy.core.multiarray import _reconstruct, scalar

    numpy_dtypes = (
        np.bool_, np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    )
    safe_globals = [
        _reconstruct,
        scalar,
        np.ndarray,
        np.dtype,
        *(type(np.dtype(dtype)) for dtype in numpy_dtypes),
    ]
    try:
        with torch.serialization.safe_globals(safe_globals):
            data = torch.load(path, map_location="cpu", weights_only=True)
    except AttributeError:  # PyTorch versions predating safe_globals.
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict) or "rewards" not in data:
        raise ValueError(f"{path} does not contain a 'rewards' entry")

    rewards = np.asarray(data["rewards"])
    if rewards.size == 0:
        raise ValueError(f"{path} contains no rewards")
    if not np.issubdtype(rewards.dtype, np.number):
        raise TypeError(f"Expected numeric rewards, got dtype {rewards.dtype}")
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    elif rewards.ndim != 2:
        raise ValueError(
            f"Expected rewards shaped [transitions, agents], got {rewards.shape}"
        )
    if not np.isfinite(rewards).all():
        raise ValueError(f"{path} contains non-finite rewards")
    return rewards


def histogram_bins(values: np.ndarray, requested: int | None):
    """Center bins on small discrete reward supports; otherwise use NumPy auto."""
    if requested is not None:
        return requested
    unique = np.unique(values)
    if unique.size == 1:
        width = max(abs(float(unique[0])) * 0.1, 0.5)
        return [unique[0] - width, unique[0] + width]
    if unique.size <= 20:
        midpoints = (unique[:-1] + unique[1:]) / 2
        return np.concatenate(
            ([unique[0] - (midpoints[0] - unique[0])], midpoints,
             [unique[-1] + (unique[-1] - midpoints[-1])])
        )
    return "auto"


def plot_rewards(
    rewards: np.ndarray,
    output: Path,
    experiment_name: str,
    bins: int | None = None,
) -> None:
    """Write a histogram containing every per-agent transition reward."""
    values = rewards.reshape(-1)
    unique = np.unique(values)
    discrete = bins is None and unique.size <= 20
    fig, axis = plt.subplots(figsize=(9, 6))
    counts, _, patches = axis.hist(
        values,
        bins=histogram_bins(values, bins),
        color="#3977A3",
        edgecolor="white",
        linewidth=0.8,
        rwidth=0.75 if discrete else None,
    )
    axis.set(
        title=f"R2BC demonstration rewards\n{experiment_name}",
        xlabel="Reward",
        ylabel="Agent-transition count",
    )
    axis.grid(axis="y", alpha=0.25)
    if discrete:
        axis.set_xticks(unique)
        for count, patch in zip(counts, patches):
            axis.annotate(
                f"{int(count):,}",
                (patch.get_x() + patch.get_width() / 2, count),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    stats = (
        f"{rewards.shape[0]:,} transitions × {rewards.shape[1]} agents\n"
        f"n = {values.size:,}   mean = {values.mean():.4g}   "
        f"std = {values.std():.4g}\n"
        f"min = {values.min():.4g}   max = {values.max():.4g}"
    )
    axis.text(
        0.98,
        0.97,
        stats,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a histogram of all per-agent transition rewards in an R2BC experiment."
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="R2BC experiment directory, or a path to demonstrations.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image (default: <experiment>/demonstration_rewards_histogram.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        help="Number of histogram bins (default: automatic, discrete-aware)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_path = demonstration_path(args.experiment)
    output = args.output or demo_path.with_name("demonstration_rewards_histogram.png")
    rewards = load_rewards(demo_path)
    plot_rewards(rewards, output, demo_path.parent.name, args.bins)
    print(f"Saved reward histogram to {output.resolve()}")


if __name__ == "__main__":
    main()
