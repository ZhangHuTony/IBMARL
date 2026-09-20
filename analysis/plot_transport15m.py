"""Plot transport15m sweep variants against each other.

Usage:
    python analysis/plot_transport15m.py                       # ibmarl vs strict, 4 panels
    python analysis/plot_transport15m.py --variants ibmarl,maddpg,rlfd --panels B --max-iters 1500
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Project-stable variant -> (label, color).  Colors are assigned once per variant
# from the validated reference palette and never reassigned, so a variant keeps
# its hue across every figure regardless of which others are drawn beside it.
# rlfd takes violet (slot 7) rather than the next fixed slot (yellow) because it
# sits on top of maddpg near zero and yellow is too low-contrast on the light
# surface for a 2px line to be told apart there.
VARIANTS = {
    "ibmarl":        ("IBMARL (per-agent mixing)",         "#2a78d6"),  # slot 1 blue
    "ibmarl_strict": ("IBMARL (strict: all-IL or all-RL)", "#eb6834"),  # slot 2 orange
    "maddpg":        ("MADDPG",                            "#1baf7a"),  # slot 3 aqua
    "rlfd":          ("RLfD",                              "#4a3aa7"),  # slot 7 violet
    "rft":           ("RFT",                               "#e87ba4"),  # slot 5 magenta
}
DEFAULT_VARIANTS = "ibmarl,ibmarl_strict"

INK = "#52514e"
GRID = "#d8d7d2"
SMOOTH = 25  # rolling window, in iterations


def load(sweep: Path, variant: str, seed: int, max_iters: int | None = None) -> pd.DataFrame:
    csv = sweep / variant / f"seed_{seed}" / "data" / "metrics.csv"
    df = pd.read_csv(csv)
    df = df[df["group"] == "agents"]
    if max_iters is not None:
        df = df[df["iteration"] < max_iters]
    return df.reset_index(drop=True)


def bc_baseline(sweep: Path, seed: int):
    csv = sweep / "bc_eval" / f"seed_{seed}" / "data" / "metrics.csv"
    if not csv.exists():
        return None
    row = pd.read_csv(csv).iloc[0]
    return float(row["eval_reward_mean"]), float(row.get("eval_reward_sem", float("nan")))


def series(ax, df, col, color, label, smooth=SMOOTH):
    """Raw trace at low alpha with a rolling mean on top."""
    x, y = df["iteration"], df[col]
    ax.plot(x, y, color=color, alpha=0.16, linewidth=0.8)
    ax.plot(x, y.rolling(smooth, min_periods=1, center=True).mean(),
            color=color, linewidth=2, label=label)


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel("Training iteration", fontsize=9, color=INK)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=8, colors=INK, length=3)


def main():
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--sweep", type=Path, default=root / "results" / "transport15m")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--variants", default=DEFAULT_VARIANTS,
                   help=f"comma-separated variants to draw; known: {','.join(VARIANTS)}")
    p.add_argument("--max-iters", type=int, default=None,
                   help="clip every run to iterations < this, for an equal-budget comparison")
    p.add_argument("--panels", default="A,B,C,D",
                   help="comma-separated panels to draw: A,B,C,D")
    args = p.parse_args()

    sweep = args.sweep
    outdir = args.outdir or sweep / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; known: {sorted(VARIANTS)}")
    runs = []
    for name in names:
        lbl, color = VARIANTS[name]
        runs.append((lbl, color, load(sweep, name, args.seed, args.max_iters)))
    bc = bc_baseline(sweep, args.seed)
    n_iters = min(len(df) for _, _, df in runs)

    panels = {
        "A": dict(col="episode_reward_mean", baseline=False, ylim=None,
                  title="Training return (executed arbiter policy)",
                  ylabel="Episode reward", slug="train_return"),
        "B": dict(col="eval_reward_mean", baseline=True, ylim=None,
                  title="Evaluation return (rl_il protocol)",
                  ylabel="Eval reward", slug="eval_return"),
        "C": dict(col="rl_only_episode_reward_mean", baseline=True, ylim=None,
                  title="RL-only return (RL actor without the teacher)",
                  ylabel="Episode reward", slug="rl_only_return"),
        "D": dict(col="rl_action_fraction", baseline=False, ylim=(0, 1),
                  title="Arbiter RL-action fraction",
                  ylabel="Fraction of actions from RL", slug="rl_action_fraction"),
    }
    keys = [k.strip().upper() for k in args.panels.split(",") if k.strip()]
    bad = [k for k in keys if k not in panels]
    if bad:
        raise SystemExit(f"unknown panel(s) {bad}; choose from {sorted(panels)}")

    single = len(keys) == 1
    ncols = 1 if single else 2
    nrows = 1 if single else (len(keys) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.5, 4.8) if single else (11, 3.75 * nrows),
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, key in zip(axes.flat, keys):
        spec = panels[key]
        ax.set_facecolor("#fcfcfb")
        drawn = 0
        for lbl, color, df in runs:
            if spec["col"] not in df.columns:
                continue  # baselines log no arbiter columns
            series(ax, df, spec["col"], color, lbl)
            drawn += 1
        if not drawn:
            raise SystemExit(
                f"panel {key} needs column {spec['col']!r}, which none of {names} logs")
        if spec["baseline"] and bc is not None:
            ax.axhline(bc[0], color=INK, linestyle="--", linewidth=1.5,
                       label=f"BC teacher ({bc[0]:.1f})")
        if key == "D":
            ax.axhline(0.5, color=GRID, linewidth=1)
        if spec["ylim"]:
            ax.set_ylim(*spec["ylim"])
        title = spec["title"] if single else f"{key}. {spec['title']}"
        style(ax, title, spec["ylabel"])
        ax.legend(fontsize=9 if single else 8, frameon=False,
                  labelcolor=INK, loc="best")

    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)

    fig.suptitle(
        f"transport15m (seed {args.seed}) — {n_iters} iterations x 5k frames "
        f"= {n_iters * 5000 / 1e6:.2f}M environment steps",
        fontsize=12.5, x=0.008, ha="left", color="#0b0b0b",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94 if single else 0.965))

    stem = "transport15m_" + "_".join(names)
    stem += f"_{panels[keys[0]]['slug']}" if single else "_all"
    for ext in ("png", "pdf"):
        out = outdir / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
        print(f"wrote {out}")

    # Terminal summary over the last 100 iterations.
    print(f"\nlast-100-iteration means (of {n_iters} iterations):")
    if bc is not None:
        print(f"  {'BC teacher':<18} eval={bc[0]:7.2f} (sem {bc[1]:.2f})")
    for lbl, _, df in runs:
        t = df.tail(100)
        line = (f"  {lbl:<36} eval={t['eval_reward_mean'].mean():7.2f}  "
                f"train={t['episode_reward_mean'].mean():7.2f}")
        if "rl_only_episode_reward_mean" in t.columns:
            line += (f"  rl_only={t['rl_only_episode_reward_mean'].mean():7.2f}"
                     f"  rl_frac={t['rl_action_fraction'].mean():.3f}")
        print(line)


if __name__ == "__main__":
    main()
