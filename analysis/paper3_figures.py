"""
Figure generation for the paper3 sweep (IROS / IEEEtran two-column conventions).

Outputs vector PDF (for \\includegraphics) and 600 dpi PNG to
analysis/figures/paper3/.

Metric.  Curves are the raw per-agent mean episode return.  The navigation task
uses the sparse-reward transform: an agent receives -1 for every step it is
farther than gt_radius from its landmark and the underlying VMAS shaping reward
(~0) once it is on the landmark.  Episodes always run the full 100-step horizon,
so the return lives in [-100, ~0]: -100 means the agent never reached its goal,
-20 means it arrived after roughly 20 steps and stayed.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "paper3"
OUT = ROOT / "analysis" / "figures" / "paper3"
OUT.mkdir(parents=True, exist_ok=True)

SMOOTH = 9          # centred rolling window, in iterations (= 1k env steps each)
PNG_DPI = 600       # IEEE wants >= 300 dpi for raster figures

# --- IEEEtran geometry (inches) -------------------------------------------
COL_W = 3.487       # \columnwidth
TEXT_W = 7.16       # \textwidth (double column)

# --- Style: Times-metric serif to match an IEEEtran body -------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.4,
    "legend.handletextpad": 0.5,
    "pdf.fonttype": 42,          # embed TrueType, required by IEEE PDF eXpress
    "ps.fonttype": 42,
    # No tight-bbox cropping: each figure is laid out at its exact IEEEtran
    # width so \\includegraphics[width=...] reproduces it 1:1 and the 8pt tick
    # labels really render at 8pt in the paper.
    "savefig.bbox": None,
})

C = {
    "ibmarl_strict":  "#E8253D",   # red   -- ours (joint teacher-vs-RL choice)
    "rlfd":           "#3B75AF",   # blue
    "rft":            "#4E9A50",   # green
    "maddpg":         "#8C6D31",   # brown
    "bc":             "#7F7F7F",   # grey dashed
    "ibmarl":         "#7B52AB",   # purple -- per-agent mixing variant
    "ibmarl_strict_hard":    "#F0912D",   # orange -- strict + argmax
    "ibmarl_strict_1critic": "#F293B4",   # pink   -- strict + single critic
    "ibmarl_hard":    "#F0912D",   # orange
    "ibmarl_1critic": "#F293B4",   # pink
}
LABEL = {
    "ibmarl_strict":  "IBMARL",
    "rlfd":           "RLfD",
    "rft":            "RFT",
    "maddpg":         "MADDPG",
    "ibmarl":         "IBMARL w/ per-agent mixing",
    "ibmarl_hard":    "IBMARL w/ mixing, argmax",
    "ibmarl_1critic": "IBMARL w/ mixing, single critic",
    "ibmarl_strict_hard":    "IBMARL w/o soft selection",
    "ibmarl_strict_1critic": "IBMARL w/o critic ensemble",
}
BASELINES = ["ibmarl_strict", "rlfd", "rft", "maddpg"]
ABLATIONS = ["ibmarl_strict", "ibmarl", "ibmarl_strict_hard", "ibmarl_strict_1critic"]

RETURN_LABEL = "Episode return"


# --------------------------------------------------------------------------
def load(variant, run="paper3"):
    frames = [
        pd.read_csv(f)
        for f in sorted(glob.glob(str(ROOT / "results" / run / variant /
                                      "seed_*" / "data" / "metrics.csv")),
                        key=lambda p: int(p.split("seed_")[1].split("/")[0]))
    ]
    if not frames:
        raise FileNotFoundError(f"{run}/{variant}")
    return frames


def curve(variant, col, smooth=SMOOTH):
    """Steps (in thousands) and the per-seed matrix of the smoothed column."""
    series, steps = [], None
    for d in load(variant):
        sub = d[["step", col]].dropna()
        series.append(sub[col].rolling(smooth, center=True, min_periods=1).mean().values)
        if steps is None:
            steps = sub["step"].values / 1000.0
    n = min(len(v) for v in series)
    return steps[:n], np.stack([v[:n] for v in series])


def mean_sem(y):
    m = y.mean(0)
    sem = y.std(0, ddof=1) / np.sqrt(y.shape[0]) if y.shape[0] > 1 else np.zeros_like(m)
    return m, sem


def band(ax, variant, col, color=None):
    x, y = curve(variant, col)
    m, sem = mean_sem(y)
    c = color or C[variant]
    ax.plot(x, m, color=c, label=LABEL[variant], zorder=3)
    ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.18, lw=0, zorder=2)


def bc_level():
    vals = [pd.read_csv(f)["eval_reward_mean"].iloc[0]
            for f in sorted(glob.glob(str(RUN / "bc_eval" / "seed_*" /
                                          "data" / "metrics.csv")))]
    return float(np.mean(vals))


def style_return_axis(ax, title=None, ylabel=True, xlabel=True):
    if title:
        ax.set_title(title, pad=4)
    ax.set_xlim(0, 300)
    ax.set_ylim(-100, 0)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([-100, -80, -60, -40, -20, 0])
    if xlabel:
        ax.set_xlabel(r"Interaction steps ($\times$1000)")
    if ylabel:
        ax.set_ylabel(RETURN_LABEL)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=PNG_DPI, facecolor="white")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {name}.pdf / .png  ({w:.2f} x {h:.2f} in, png {PNG_DPI} dpi)")


# ============================== Figure 1 ==================================
def fig_main():
    """Evaluation learning curves: IBMARL against the three baselines."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for v in BASELINES:
        band(ax, v, "eval_reward_mean")
    ax.axhline(bc_level(), color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
               label="R2BC (teacher)")
    style_return_axis(ax)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2, labelspacing=0.32)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    save(fig, "fig1_main")


# ============================== Figure 2 ==================================
def fig_ablation():
    """Ablation learning curves, single-column, matching fig1's layout."""
    short = {
        "ibmarl_strict":         "IBMARL",
        "ibmarl":                "w/ per-agent mixing",
        "ibmarl_strict_hard":    "w/o soft selection",
        "ibmarl_strict_1critic": "w/o critic ensemble",
    }
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for v in ABLATIONS:
        x, y = curve(v, "eval_reward_mean")
        m, sem = mean_sem(y)
        ax.plot(x, m, color=C[v], label=short[v], zorder=3)
        ax.fill_between(x, m - sem, m + sem, color=C[v], alpha=0.18, lw=0, zorder=2)
    ax.axhline(bc_level(), color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
               label="R2BC (teacher)")
    style_return_axis(ax)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2, labelspacing=0.32)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    save(fig, "fig2_ablation")


# ============================== Figure 3 ==================================
def fig_seeds():
    """Per-seed final return, i.e. run-to-run reliability."""
    bc = bc_level()
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    order = BASELINES + ABLATIONS[1:]
    for i, v in enumerate(order):
        finals = np.array([d["eval_reward_mean"].dropna().tail(10).mean()
                           for d in load(v)])
        ax.scatter(np.full(len(finals), i) +
                   (np.arange(len(finals)) - (len(finals) - 1) / 2) * 0.09,
                   finals, s=11, color=C[v], zorder=3, linewidths=0)
        ax.hlines(finals.mean(), i - 0.28, i + 0.28, color=C[v], lw=1.6, zorder=2)
    ax.axhline(bc, color=C["bc"], ls=(0, (4, 2.5)), lw=1.0, zorder=1)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.text(len(order) - 0.5, bc + 1.2, "R2BC", fontsize=7, color=C["bc"], ha="right")
    ax.axvline(3.5, color="0.8", lw=0.6, zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["IBMARL", "RLfD", "RFT", "MADDPG",
                        "w/ mix", "w/o soft", "w/o ens."], rotation=25, ha="right",
                       fontsize=6.8)
    ax.tick_params(axis="x", pad=1)
    ax.set_ylim(-65, -15)
    ax.set_yticks([-60, -50, -40, -30, -20])
    ax.set_ylabel(f"Final {RETURN_LABEL.lower()}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.965, bottom=0.235)
    save(fig, "fig3_seed_spread")


# ============================== Figure 4 ==================================
def fig_replication():
    """
    The ablation ordering does not replicate across sweeps.

    Both sweeps' *final* checkpoints re-measured under one identical, correct
    evaluate() at 500 episodes/seed (results/<tag>/checkpoint_eval_fixed.csv,
    produced by analysis/eval_checkpoints.py).  Nothing here depends on the
    training-time logged metric, so the pre-fix and post-fix runs are directly
    comparable.
    """
    order = ABLATIONS
    short = {"ibmarl": "full", "ibmarl_strict": "w/o mix",
             "ibmarl_hard": "w/o soft", "ibmarl_1critic": "w/o ens."}
    tags = [("paper", "none"), ("paper3", "fill")]

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    for i, v in enumerate(order):
        pooled = []
        for j, (tag, style) in enumerate(tags):
            df = pd.read_csv(ROOT / "results" / tag / "checkpoint_eval_fixed.csv")
            y = df[df.variant == v]["eval_reward_mean_fixed"].values
            pooled.append(y)
            x = i + (j - 0.5) * 0.30
            if style == "none":
                ax.scatter(np.full(len(y), x), y, s=13, zorder=3, linewidths=0.9,
                           facecolors="none", edgecolors=C[v])
            else:
                ax.scatter(np.full(len(y), x), y, s=13, zorder=3, linewidths=0,
                           color=C[v])
        allv = np.concatenate(pooled)
        ax.hlines(allv.mean(), i - 0.34, i + 0.34, color=C[v], lw=1.6, zorder=2)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[v] for v in order], fontsize=7.5)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-30, -15)
    ax.set_yticks([-30, -27, -24, -21, -18, -15])
    ax.set_ylabel(f"Final {RETURN_LABEL.lower()}")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.6, markerfacecolor="none",
               markeredgecolor="0.35", label="pre-fix sweep"),
        Line2D([], [], marker="o", ls="none", ms=3.6, color="0.35",
               label="post-fix sweep"),
        Line2D([], [], color="0.35", lw=1.6, label="pooled mean"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, ncol=1,
              borderpad=0.2, labelspacing=0.35)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.965, bottom=0.135)
    save(fig, "fig4_replication")


if __name__ == "__main__":
    fig_main()
    fig_ablation()
    fig_seeds()
    fig_replication()
