"""
Before/after comparison for the demo-loader and critic-ensemble bug fixes.

Arms (see results/bugfix_compare/<arm>/, each holding one subdir per seed):

    A  rlfd_before    pre-fix
    B  rlfd_after     + fix #1 (demo loader reads next_obs)
    C  ibmarl_before  pre-fix
    D  ibmarl_after   + fix #1
    E  ibmarl_ensemble  + fix #1 + fix #2 (all critic ensemble members trained)

B - A isolates fix #1 (RLfD has no arbiter and no ensemble).
D - C isolates fix #1 under IBMARL; E - D is the increment from fix #2.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main_results import PROJECT_ROOT, load_metrics_for_method

# Moving-average window for the mean curve (1 = no smoothing)
N_WINDOW = 8

ROOT = "results/bugfix_compare"

# panel title -> [(label, arm dir, colour), ...]
PANELS: dict[str, list[tuple[str, str, str]]] = {
    "RLfD — isolates fix #1": [
        ("before (next_obs == obs)", f"{ROOT}/rlfd_before", "tab:red"),
        ("after fix #1", f"{ROOT}/rlfd_after", "tab:blue"),
    ],
    "IBMARL — fix #1 then fix #2": [
        ("before", f"{ROOT}/ibmarl_before", "tab:red"),
        ("+ fix #1 (next_obs)", f"{ROOT}/ibmarl_after", "tab:blue"),
        ("+ fix #2 (full ensemble)", f"{ROOT}/ibmarl_ensemble", "tab:green"),
    ],
}

# metric -> (axis label, only-for-panel-substring or None)
METRICS: dict[str, tuple[str, str | None]] = {
    "eval_reward_mean": ("Eval episode reward", None),
    "critic_loss": ("Critic loss", None),
    "rl_action_fraction": ("RL action fraction", "IBMARL"),
}


def smooth(iterations, mean, stderr):
    if N_WINDOW <= 1:
        return iterations, mean, stderr
    kernel = np.ones(N_WINDOW) / N_WINDOW
    return (
        iterations[N_WINDOW - 1:],
        np.convolve(mean, kernel, mode="valid"),
        stderr[N_WINDOW - 1:],
    )


def main() -> None:
    rows = list(METRICS.items())
    cols = list(PANELS.items())
    fig, axs = plt.subplots(
        len(rows), len(cols),
        figsize=(7 * len(cols), 4 * len(rows)),
        squeeze=False,
    )

    for r, (metric, (ylabel, only_panel)) in enumerate(rows):
        for c, (panel_title, arms) in enumerate(cols):
            ax = axs[r][c]

            if only_panel is not None and only_panel not in panel_title:
                ax.axis("off")
                continue

            plotted = False
            for label, arm_dir, colour in arms:
                if not (PROJECT_ROOT / arm_dir).is_dir():
                    continue
                try:
                    it, mean, stderr = load_metrics_for_method(arm_dir, metric=metric)
                except (FileNotFoundError, KeyError) as e:
                    print(f"  skip {arm_dir} [{metric}]: {e}")
                    continue
                it, mean, stderr = smooth(it, mean, stderr)
                ax.plot(it, mean, label=label, color=colour)
                ax.fill_between(it, mean - stderr, mean + stderr, color=colour, alpha=0.2)
                plotted = True

            ax.set_title(f"{panel_title}\n{ylabel}" if r == 0 else ylabel)
            ax.set_xlabel("Training iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if plotted:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "no runs found", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

    fig.suptitle("Bug-fix before/after (mean ± stderr across seeds)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = PROJECT_ROOT / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig_bugfix_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    summarise()


def summarise() -> None:
    """Print final-value deltas so the effect size is readable without the plot."""
    print("\nFinal-100-iteration mean of eval_reward_mean:")
    baselines = {"rlfd": f"{ROOT}/rlfd_before", "ibmarl": f"{ROOT}/ibmarl_before"}
    finals: dict[str, float] = {}

    for _, arms in PANELS.items():
        for label, arm_dir, _ in arms:
            if not (PROJECT_ROOT / arm_dir).is_dir():
                continue
            try:
                _, mean, _ = load_metrics_for_method(arm_dir, metric="eval_reward_mean")
            except (FileNotFoundError, KeyError):
                continue
            tail = mean[-100:] if len(mean) > 100 else mean
            finals[arm_dir] = float(np.nanmean(tail))
            print(f"  {arm_dir:45s} {finals[arm_dir]:8.3f}   ({label})")

    print("\nDeltas vs baseline:")
    for arm_dir, value in finals.items():
        base_key = next((b for k, b in baselines.items() if k in arm_dir), None)
        if base_key and base_key in finals and arm_dir != base_key:
            print(f"  {arm_dir:45s} {value - finals[base_key]:+8.3f}")


if __name__ == "__main__":
    main()
