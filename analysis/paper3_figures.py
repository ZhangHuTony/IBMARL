"""
Figure generation for a paper sweep (IROS / IEEEtran two-column conventions).

    python -m analysis.paper3_figures                 # the paper3 nav sweep
    python -m analysis.paper3_figures --tag buzzwire1 # the buzz_wire sweep

Outputs vector PDF (for \\includegraphics) and 600 dpi PNG to
analysis/figures/<tag>/.

Metric.  Curves are the raw per-agent mean episode return.  Both tasks use a
sparse reward: -1 for every step off-goal (agent-to-landmark for navigation,
ball-to-goal for buzz_wire) and the near-zero native VMAS reward once on it.
Episodes always run the full horizon (100 steps nav / 200 buzz_wire), so the
return reads -(steps spent off-goal): the floor (-100 / -200) means the goal
was never reached.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

# Per-tag axis presets.  A tag not listed here gets `None`s, which mean
# "let matplotlib autoscale" in style_return_axis / fig_seeds.
PRESETS = {
    "paper3": dict(
        xlim=(0, 300), xticks=[0, 100, 200, 300],
        ylim=(-100, 0), yticks=[-100, -80, -60, -40, -20, 0],
        seeds_ylim=(-65, -15), seeds_yticks=[-60, -50, -40, -30, -20],
        smooth=9,
    ),
    "buzzwire1": dict(
        xlim=(0, 600), xticks=[0, 200, 400, 600],
        ylim=(-200, 0), yticks=[-200, -150, -100, -50, 0],
        seeds_ylim=None, seeds_yticks=None,
        # buzz_wire evaluates every 4th of 75 iterations (20 evals over 600k
        # steps), not navigation's 300. A 9-eval window would span 270k steps
        # (45% of the run); 3 keeps the window at ~90k, closer to paper3's 3%.
        smooth=3,
    ),
}
# buzzwire2 (collector fix + teacher warm-up; baselines symlinked from
# buzzwire1) has the same budget and eval cadence, so the same axes and window.
PRESETS["buzzwire2"] = dict(PRESETS["buzzwire1"])
# buzzwire3 (12-demo teacher at -137.5 instead of buzzwire2's 16-demo -86.4;
# everything else identical) -- same budget and cadence again.
PRESETS["buzzwire3"] = dict(PRESETS["buzzwire1"])
# buzzwire3_reg (gated imitation term on top of buzzwire3's ibmarl_strict, the
# baseline and bc_eval symlinked from buzzwire3) -- same budget and cadence.
PRESETS["buzzwire3_reg"] = dict(PRESETS["buzzwire1"])
# buzzwire4 (binary_terminal_reward, 6-demo teacher; ibmarl_gated_a0.4 is the
# headline method as of paper_run.py's 2026-09-20 registration) -- same 600k
# budget and eval_interval, but returns are success rates in [0, 1] under the
# binary terminal schema, not the legacy -1/step return the axes above are
# scaled for.  Directory layout matches the others via symlinks (see
# results/buzzwire4/README or the run that created them): the SLURM array's
# per-task tags (buzzwire4b_<variant>/<variant>/seed_N, buzzwire4_maddpg/maddpg)
# are symlinked to results/buzzwire4/<variant> the same way buzzwire2/3
# symlink their baselines from buzzwire1.
PRESETS["buzzwire4"] = dict(
    xlim=(0, 600), xticks=[0, 200, 400, 600],
    ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1.0],
    seeds_ylim=(0, 1.05), seeds_yticks=[0, 0.25, 0.5, 0.75, 1.0],
    smooth=3,
    ylabel="Success rate",
    bc_label_dy=0.03,
)
# buzzwire5 (detuned-demonstrator teacher, eval every 2nd iteration at 200
# episodes, 5 seeds; baselines with a target actor, 400k replay for all, RFT
# annealed over the full run) -- same budget and success-rate axes.  38 evals
# instead of 19, so a 3-eval window is now ~48k steps (8% of the run).
PRESETS["buzzwire5"] = dict(PRESETS["buzzwire4"])
PRESETS["quick_results"] = dict(
    xlim=(0, 7000), xticks=[0, 2000, 4000, 6000],
    ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1.0],
    seeds_ylim=(0, 1.05), seeds_yticks=[0, 0.25, 0.5, 0.75, 1.0],
    smooth=9, ylabel="Success rate",
)
_EMPTY_PRESET = dict(xlim=None, xticks=None, ylim=None, yticks=None,
                     seeds_ylim=None, seeds_yticks=None, smooth=9, ylabel=None)

# Set by configure(); defaults preserve the historical single-purpose script.
TAG = "paper3"
RUN = ROOT / "results" / TAG
OUT = ROOT / "analysis" / "figures" / TAG
PRESET = PRESETS[TAG]

SMOOTH = 9          # centred rolling window, in iterations (1 eval per iter)
SUFFIX = ""         # extra filename tag, set by --exclude for partial figures
EXCLUDED: list = []  # variants dropped by --exclude, annotated onto fig_main
PNG_DPI = 600       # IEEE wants >= 300 dpi for raster figures

# Which evaluation protocol the logical column "eval_reward_mean" maps onto.
#   rl        -- the RL-actor protocol for every run.  Baselines and IBMARL runs
#                predating the RL+IL evaluation logged it as eval_reward_mean;
#                later IBMARL runs (eval_protocol == "rl_il") log it as
#                rl_only_episode_reward_mean.  Comparable across everything on
#                disk and the steps-to-teacher protocol.  The default, so the
#                paper's outputs are unchanged.
#   executed  -- each method's executed policy: eval_reward_mean as logged, i.e.
#                the arbitrated RL+IL policy for post-change IBMARL runs (IBRL's
#                reporting convention) and the RL actor for everything else.
PROTOCOL = "rl"


def configure(tag: str, smooth: int | None = None,
              protocol: str | None = None) -> None:
    global TAG, RUN, OUT, PRESET, SMOOTH, PROTOCOL
    TAG = tag
    RUN = ROOT / "results" / tag
    OUT = ROOT / "analysis" / "figures" / tag
    PRESET = PRESETS.get(tag, _EMPTY_PRESET)
    # An explicit --smooth wins; otherwise take the tag's preset, whose window
    # is sized against that tag's evaluation cadence.
    SMOOTH = smooth if smooth is not None else PRESET.get("smooth", 9)
    if protocol is not None:
        if protocol not in ("rl", "executed"):
            raise ValueError(f"protocol must be 'rl' or 'executed', got {protocol!r}")
        PROTOCOL = protocol
    variant_set = VARIANT_SETS.get(tag)
    BASELINES[:] = variant_set["baselines"] if variant_set else _DEFAULT_BASELINES
    ABLATIONS[:] = variant_set["ablations"] if variant_set else _DEFAULT_ABLATIONS

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
    # actor-lag experiment: IBMARL + gated imitation term, by alpha
    "ibmarl_strict_gated_a0.1": "#66C2A5",   # light teal
    "ibmarl_strict_gated_a0.4": "#1B9E77",   # teal
    "ibmarl_strict_gated_a1.6": "#0B5D47",   # dark teal
    # buzzwire4: mixing + gated term is the headline method now (paper_run.py,
    # 2026-09-20), so it takes over the "ours" red from ibmarl_strict.
    "ibmarl_gated_a0.4":          "#E8253D",   # red -- ours
    "ibmarl_gated_a0.4_1critic":  "#F293B4",   # pink -- w/o critic ensemble
    "ibmarl_regularizer":          "#E8253D",
    "isolated_ibmarl":              "#3B75AF",
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
    "ibmarl_strict_gated_a0.1": r"IBMARL + gated term ($\alpha$=0.1)",
    "ibmarl_strict_gated_a0.4": r"IBMARL + gated term ($\alpha$=0.4)",
    "ibmarl_strict_gated_a1.6": r"IBMARL + gated term ($\alpha$=1.6)",
    "ibmarl_gated_a0.4":          "IBMARL",
    "ibmarl_gated_a0.4_1critic":  "IBMARL w/o critic ensemble",
    "ibmarl_regularizer":          "IBMARL + regularizer",
    "isolated_ibmarl":              "IBMARL",
}
# Tick labels for fig_seeds' crowded categorical axis, where LABEL is too long.
SHORT = {
    "ibmarl_strict":  "IBMARL",
    "rlfd":           "RLfD",
    "rft":            "RFT",
    "maddpg":         "MADDPG",
    "ibmarl":         "w/ mix",
    "ibmarl_strict_hard":    "w/o soft",
    "ibmarl_strict_1critic": "w/o ens.",
    "ibmarl_strict_gated_a0.1": "gated .1",
    "ibmarl_strict_gated_a0.4": "gated .4",
    "ibmarl_strict_gated_a1.6": "gated 1.6",
    "ibmarl_gated_a0.4":          "IBMARL",
    "ibmarl_gated_a0.4_1critic":  "w/o ens.",
    "ibmarl_regularizer": "regularizer",
    "isolated_ibmarl": "isolated",
}
# fig_seeds' x-tick labels, overridden per tag where SHORT's existing entry was
# worded for a different sweep's framing ("ibmarl"/"ibmarl_strict_gated_a0.4"
# read as "w/ mix"/"gated .4" from buzzwire3_reg's alpha-sweep angle; buzzwire4
# ablates off the gated method, so "w/o gate"/"w/o mixing" is what the same
# variant means here).  A tag or variant not listed here keeps SHORT's entry.
SHORT_OVERRIDES = {
    "buzzwire4": {"ibmarl": "w/o gate", "ibmarl_strict_gated_a0.4": "w/o mixing"},
}
SHORT_OVERRIDES["buzzwire5"] = SHORT_OVERRIDES["buzzwire4"]
_DEFAULT_BASELINES = ["ibmarl_strict", "rlfd", "rft", "maddpg"]
_DEFAULT_ABLATIONS = ["ibmarl_strict", "ibmarl", "ibmarl_strict_hard", "ibmarl_strict_1critic"]
# Per-tag variant sets, for a sweep whose headline method has a different name
# (buzzwire4's is ibmarl_gated_a0.4, not ibmarl_strict).  A tag not listed here
# gets _DEFAULT_BASELINES/_DEFAULT_ABLATIONS, so every existing tag's output is
# unchanged.  Applied by configure(), same as PRESETS.
VARIANT_SETS = {
    "buzzwire4": dict(
        baselines=["ibmarl_gated_a0.4", "rlfd", "rft", "maddpg"],
        # Main method first, matching every other tag's convention that
        # ABLATIONS[0] == BASELINES[0] -- fig_seeds' ABLATIONS[1:] depends on
        # it to avoid plotting the main method's column twice.
        ablations=["ibmarl_gated_a0.4", "ibmarl",
                   "ibmarl_strict_gated_a0.4", "ibmarl_gated_a0.4_1critic"],
    ),
}
VARIANT_SETS["buzzwire5"] = VARIANT_SETS["buzzwire4"]
VARIANT_SETS["quick_results"] = dict(
    baselines=["ibmarl_regularizer", "isolated_ibmarl"],
    # The seed-spread figure appends ABLATIONS[1:] to BASELINES, so keep the
    # quick comparison's two trial sets in the baseline block only.
    ablations=["ibmarl_regularizer"],
)
BASELINES = list(_DEFAULT_BASELINES)
ABLATIONS = list(_DEFAULT_ABLATIONS)
# fig_ablation()'s legend labels drop the "IBMARL" prefix and name the removed
# ingredient; what's removed differs by tag (buzzwire1-3 ablate mixing/soft
# selection/critic count off ibmarl_strict, buzzwire4 ablates the gated term/
# mixing/critic count off ibmarl_gated_a0.4), so this is keyed by tag rather
# than reused from module-level SHORT (which serves fig_seeds' unrelated
# crowded x-axis).
_DEFAULT_ABLATION_SHORT = {
    "ibmarl_strict":         "IBMARL",
    "ibmarl":                "w/ per-agent mixing",
    "ibmarl_strict_hard":    "w/o soft selection",
    "ibmarl_strict_1critic": "w/o critic ensemble",
}
ABLATION_SHORT = {
    "buzzwire4": {
        "ibmarl":                    "w/o gated term",
        "ibmarl_gated_a0.4":         "IBMARL",
        "ibmarl_strict_gated_a0.4":  "w/o mixing",
        "ibmarl_gated_a0.4_1critic": "w/o critic ensemble",
    },
}
ABLATION_SHORT["buzzwire5"] = ABLATION_SHORT["buzzwire4"]
ABLATION_SHORT["quick_results"] = {
    "ibmarl_regularizer": "IBMARL + regularizer",
    "isolated_ibmarl": "isolated IBMARL",
}

RETURN_LABEL = "Episode return"


# --------------------------------------------------------------------------
def load(variant, run=None):
    run = run or TAG
    if run == "quick_results" and variant == "ibmarl_regularizer":
        # This quick run has one seed, unlike the regular multi-seed layout.
        path = ROOT / "results" / run / variant / "seed_0" / "data" / "metrics.csv"
        if path.exists():
            return [pd.read_csv(path)]
    frames = [
        pd.read_csv(f)
        for f in sorted(glob.glob(str(ROOT / "results" / run / variant /
                                      "seed_*" / "data" / "metrics.csv")),
                        key=lambda p: int(p.split("seed_")[1].split("/")[0]))
    ]
    if not frames:
        raise FileNotFoundError(f"{run}/{variant}")
    return frames


def resolve_col(df, col):
    """
    Map the logical evaluation column onto the run's logged columns for the
    active PROTOCOL.  Only "eval_reward_mean" is protocol-dependent, and only
    for runs that logged eval_protocol == "rl_il" (IbmarlExperiment after the
    RL+IL evaluation was added); every other frame is returned unchanged.
    """
    if col != "eval_reward_mean" or "eval_protocol" not in df.columns:
        return col
    logged = df["eval_protocol"].dropna()
    if logged.empty or str(logged.iloc[0]) != "rl_il":
        return col
    return "rl_only_episode_reward_mean" if PROTOCOL == "rl" else col


def curve(variant, col, smooth=None):
    """
    Steps (in thousands) and the per-seed matrix of the smoothed column.

    Seeds are truncated to the shortest one, so a still-running or crashed seed
    silently shortens every curve for that variant -- and, for steps-to-teacher,
    drags the "training budget" down with it.  Ragged lengths are therefore
    reported loudly rather than absorbed.
    """
    smooth = smooth or SMOOTH
    series, steps = [], None
    for d in load(variant):
        c = resolve_col(d, col)
        sub = d[["step", c]].dropna()
        series.append(sub[c].rolling(smooth, center=True, min_periods=1).mean().values)
        if steps is None:
            steps = sub["step"].values / 1000.0
    lengths = [len(v) for v in series]
    n = min(lengths)
    if n != max(lengths):
        print(f"  !! {variant}: ragged seed lengths {lengths} -- truncating all "
              f"to {n} evals. Incomplete runs make these numbers meaningless.")
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
    if TAG == "quick_results":
        raise FileNotFoundError("quick_results has no bc_eval metrics")
    vals = [pd.read_csv(f)["eval_reward_mean"].iloc[0]
            for f in sorted(glob.glob(str(RUN / "bc_eval" / "seed_*" /
                                          "data" / "metrics.csv")))]
    return float(np.mean(vals))


def style_return_axis(ax, title=None, ylabel=True, xlabel=True):
    if title:
        ax.set_title(title, pad=4)
    if PRESET["xlim"]:
        ax.set_xlim(*PRESET["xlim"])
    if PRESET["ylim"]:
        ax.set_ylim(*PRESET["ylim"])
    if PRESET["xticks"]:
        ax.set_xticks(PRESET["xticks"])
    if PRESET["yticks"]:
        ax.set_yticks(PRESET["yticks"])
    if xlabel:
        ax.set_xlabel(r"Interaction steps ($\times$1000)")
    if ylabel:
        ax.set_ylabel(PRESET.get("ylabel") or RETURN_LABEL)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    # A non-default protocol gets its own files, so it never overwrites the
    # RL-actor-protocol figures the paper was built from.  SUFFIX does the same
    # for a partial figure (see --exclude): a figure missing a method must never
    # land on the filename of the complete one.
    if PROTOCOL != "rl":
        name = f"{name}_{PROTOCOL}"
    name = f"{name}{SUFFIX}"
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=PNG_DPI, facecolor="white")
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {name}.pdf / .png  ({w:.2f} x {h:.2f} in, png {PNG_DPI} dpi)")


# ============================== Figure 1 ==================================
def fig_main():
    """Evaluation learning curves: IBMARL against the three baselines."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    missing = []
    for v in BASELINES:
        # Same tolerance fig_ablation/fig_seeds already have, so a sweep still
        # in flight can be inspected mid-run.  Loud, because a baseline silently
        # absent from THIS figure would misrepresent the main comparison.
        try:
            band(ax, v, "eval_reward_mean")
        except FileNotFoundError:
            missing.append(v)
            print(f"  .. fig_main: skipping {v} (not in this sweep)")
    if TAG == "quick_results":
        final_path = RUN / "isolated_ibmarl" / "final_eval.csv"
        if final_path.exists():
            final = pd.read_csv(final_path)["eval_reward_mean_rl"].dropna().to_numpy()
            x = PRESET["xlim"][1]
            m, sem = mean_sem(final[:, None])
            ax.errorbar(x, m[0], yerr=sem[0], fmt="o", ms=4.5,
                        color=C["isolated_ibmarl"], capsize=2.5,
                        label=LABEL["isolated_ibmarl"], zorder=4)
    # Anything absent -- whether it has no runs yet or was dropped by --exclude --
    # is stated on the figure itself.  The filename suffix is not enough: figures
    # get pasted into slides and issues without it.
    absent = missing + [v for v in EXCLUDED if v not in missing]
    if absent:
        ax.text(0.015, 0.03, "incomplete: no " + ", ".join(LABEL[v] for v in absent),
                transform=ax.transAxes, fontsize=5.5,
                color="#B03030", va="bottom", ha="left", zorder=5)
    try:
        bc = bc_level()
    except FileNotFoundError:
        bc = None
    if bc is not None:
        ax.axhline(bc, color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
                   label="R2BC (teacher)")
    style_return_axis(ax)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2, labelspacing=0.32)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    save(fig, "fig1_main")


# ============================== Figure 2 ==================================
def fig_ablation():
    """Ablation learning curves, single-column, matching fig1's layout."""
    short = ABLATION_SHORT.get(TAG, _DEFAULT_ABLATION_SHORT)
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    for v in ABLATIONS:
        # A sweep need not carry every ablation (buzzwire2 runs only the two
        # main IBMARL variants); plot what is on disk rather than aborting the
        # whole figure run.  No-op for a full sweep such as paper3.
        try:
            x, y = curve(v, "eval_reward_mean")
        except FileNotFoundError:
            print(f"  .. fig_ablation: skipping {v} (not in this sweep)")
            continue
        m, sem = mean_sem(y)
        ax.plot(x, m, color=C[v], label=short[v], zorder=3)
        ax.fill_between(x, m - sem, m + sem, color=C[v], alpha=0.18, lw=0, zorder=2)
    try:
        bc = bc_level()
    except FileNotFoundError:
        bc = None
    if bc is not None:
        ax.axhline(bc, color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
                   label="R2BC (teacher)")
    style_return_axis(ax)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2, labelspacing=0.32)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    save(fig, "fig2_ablation")


# ============================== Figure 3 ==================================
def fig_seeds():
    """Per-seed final return, i.e. run-to-run reliability."""
    try:
        bc = bc_level()
    except FileNotFoundError:
        bc = None
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    order = BASELINES + ABLATIONS[1:]
    for i, v in enumerate(order):
        try:
            frames = load(v)
        except FileNotFoundError:
            print(f"  .. fig_seeds: skipping {v} (not in this sweep)")
            continue
        finals = np.array([d[resolve_col(d, "eval_reward_mean")].dropna().tail(10).mean()
                           for d in frames])
        ax.scatter(np.full(len(finals), i) +
                   (np.arange(len(finals)) - (len(finals) - 1) / 2) * 0.09,
                   finals, s=11, color=C[v], zorder=3, linewidths=0)
        ax.hlines(finals.mean(), i - 0.28, i + 0.28, color=C[v], lw=1.6, zorder=2)
    if bc is not None:
        ax.axhline(bc, color=C["bc"], ls=(0, (4, 2.5)), lw=1.0, zorder=1)
    ax.set_xlim(-0.6, len(order) - 0.4)
    # Offset sized per-preset: 1.2 (the historical default) reads fine against
    # the legacy return scale (tens of units) but would push the label off a
    # [0, 1]-ranged success-rate axis invisibly, with no error to flag it.
    if bc is not None:
        ax.text(len(order) - 0.5, bc + PRESET.get("bc_label_dy", 1.2), "R2BC",
                fontsize=7, color=C["bc"], ha="right")
    # Divider between the baseline block and the ablation block, and tick labels,
    # both derived from `order` -- a hardcoded 7 labels crashed the figure as
    # soon as --exclude or a missing ablation shortened it.
    ax.axvline(len(BASELINES) - 0.5, color="0.8", lw=0.6, zorder=0)
    overrides = SHORT_OVERRIDES.get(TAG, {})
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([overrides.get(v, SHORT[v]) for v in order], rotation=25,
                       ha="right", fontsize=6.8)
    ax.tick_params(axis="x", pad=1)
    if PRESET["seeds_ylim"]:
        ax.set_ylim(*PRESET["seeds_ylim"])
    if PRESET["seeds_yticks"]:
        ax.set_yticks(PRESET["seeds_yticks"])
    ax.set_ylabel(f"Final {(PRESET.get('ylabel') or RETURN_LABEL).lower()}")
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
    # Deliberately NOT the module-level ABLATIONS.  That list names the paper's
    # strict-based ablations (ibmarl_strict_hard / ibmarl_strict_1critic); the
    # two checkpoint_eval_fixed.csv files this figure reads predate that naming
    # and contain the mixing-based set below.  Iterating ABLATIONS here raised
    # KeyError on the label lookup and had broken this figure since the rename.
    order = ["ibmarl", "ibmarl_strict", "ibmarl_hard", "ibmarl_1critic"]
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="paper3",
                    help="results/<tag> sweep to plot (axis preset per PRESETS)")
    ap.add_argument("--smooth", type=int, default=None,
                    help=f"centred rolling window in evals (default {SMOOTH})")
    ap.add_argument("--protocol", choices=["rl", "executed"], default="rl",
                    help="rl: RL-actor evaluation for every run (default); "
                         "executed: each method's executed policy, i.e. the "
                         "arbitrated RL+IL policy for IBMARL runs that logged it")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated variants to omit entirely, e.g. a "
                         "variant still mid-sweep whose ragged seeds would "
                         "otherwise truncate every curve to its shortest one. "
                         "Output names get a _no-<variant> suffix so a partial "
                         "figure can never overwrite a complete one.")
    args = ap.parse_args()
    configure(args.tag, args.smooth, args.protocol)
    if args.exclude:
        drop = {v.strip() for v in args.exclude.split(",") if v.strip()}
        unknown = drop - set(BASELINES) - set(ABLATIONS)
        if unknown:
            raise SystemExit(f"--exclude: unknown variant(s) {sorted(unknown)}")
        BASELINES[:] = [v for v in BASELINES if v not in drop]
        ABLATIONS[:] = [v for v in ABLATIONS if v not in drop]
        SUFFIX = "_no-" + "-".join(sorted(drop))
        EXCLUDED = sorted(drop)
        print(f"excluding {EXCLUDED}")

    fig_main()
    fig_ablation()
    fig_seeds()
    if TAG == "paper3":
        # Nav-specific pre-fix/post-fix comparison; needs both sweeps'
        # checkpoint_eval_fixed.csv.
        fig_replication()
