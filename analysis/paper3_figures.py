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
# baselines and bc_eval symlinked from buzzwire3) -- same budget and cadence.
# Deliberately NOT the raw curves buzzwire5/transport800 use: this tag's figure
# sits beside buzzwire3's, whose arms it shares (baselines, bc_eval and
# ibmarl_strict are symlinks into buzzwire3), so it takes buzzwire3's preset
# unchanged -- same axes and the same 3-eval window -- and the two tags'
# baseline curves are then identical line for line.  19 evals 32k apart here,
# so 3 spans 64k; unsmoothed this cadence is step-like.
PRESETS["buzzwire3_reg"] = dict(PRESETS["buzzwire3"])
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
# annealed over the full run) -- same budget and success-rate axes.  The eval
# cadence doubled: 36 evals 16k steps apart, against buzzwire1-4's 19 at 32k.
#
# smooth=1 -- no smoothing, by request (2026-09-22): the plotted line is the raw
# across-seed mean at each evaluation.  Every earlier tag keeps its window, so
# these curves are noisier than paper3's or buzzwire1-4's by construction, not
# because this sweep is noisier.  A 200-episode success rate over 5 seeds moves
# several points eval to eval, so read the level, not the wiggles.
PRESETS["buzzwire5"] = dict(PRESETS["buzzwire4"], smooth=1)
# transport800 (4M-step budget, 201 evals 20k apart, 5 seeds per arm, same
# eight arms as buzzwire5).  Unlike buzzwire4/5 this task still runs the legacy
# -1/step sparse schema (no binary_terminal_reward in its environment yaml, see
# config/experiments/ibmarl_transport.yaml), so returns are returns, not success
# rates, and the default RETURN_LABEL/bc_label_dy apply.  On this task the
# return happens to be positive and runs 0..~24.
#
# smooth=21 spans 400k steps = 10.0% of the run, the same fraction of the budget
# buzzwire3_reg's 3-eval window covers (64k of 600k = 10.7%), so the two tags'
# curves carry comparable smoothing despite this one logging 201 evals to that
# one's 19.  Matching the eval COUNT instead would barely smooth this tag at
# all; matching the fraction of the run is what makes them look alike.
PRESETS["transport800"] = dict(
    xlim=(0, 4000), xticks=[0, 1000, 2000, 3000, 4000],
    # Sized for the smooth=21 mean +- sem envelope (-1.9 .. 23.3).  Was (-8, 27)
    # while this tag ran unsmoothed, whose raw dips reached -6.7.
    ylim=(-4, 25), yticks=[0, 5, 10, 15, 20, 25],
    seeds_ylim=(-2, 24), seeds_yticks=[0, 5, 10, 15, 20],
    smooth=21,
    # buzzwire's curves plateau early and free up the bottom-right; transport's
    # ramp is still climbing at 4M and runs straight through it, while nothing
    # reaches the top-left corner before 1.5M steps.
    legend_loc="upper left",
    # Two columns (variants | line types), so the block is 4 rows instead of 7
    # and clears the teacher line at 15.7.  The top-left corner has the width
    # to spare; buzzwire's bottom-right corner does not, hence the default 1.
    legend_ncol=2,
)
# Poster tags (2026-09-24), legacy schema: paper5 = paper3's baselines and
# teacher + paper4's ibmarl (symlinked) + the gated arms run from the frozen
# paper4 base (config/legacy/navigation); buzzwire6 = buzzwire3's arms +
# buzzwire3_reg's strict-gated arm (symlinked) + ibmarl_gated_a0.4 and the
# human-teacher arms run from the frozen buzzwire3 base.  Same budgets, teachers
# and eval cadences as the tags they extend, so they take those presets whole.
PRESETS["paper5"] = dict(PRESETS["paper3"])
PRESETS["buzzwire6"] = dict(PRESETS["buzzwire3"])
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
    # buzzwire6 human-teacher arms: same method colours as their heuristic-
    # teacher twins, told apart by the tag / panel they appear in.
    "ibmarl_gated_a0.4_human":    "#E8253D",
    "rlfd_human":                 "#3B75AF",
    "rft_human":                  "#4E9A50",
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
    "ibmarl_gated_a0.4_human":    "IBMARL (human teacher)",
    "rlfd_human":                 "RLfD (human demos)",
    "rft_human":                  "RFT (human demos)",
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
    "ibmarl_gated_a0.4_human":    "IBMARL (h)",
    "rlfd_human":                 "RLfD (h)",
    "rft_human":                  "RFT (h)",
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
SHORT_OVERRIDES["buzzwire3_reg"] = {"ibmarl_strict_gated_a0.4": "IBMARL",
                                    "ibmarl_strict": "w/o gated term"}
SHORT_OVERRIDES["transport800"] = SHORT_OVERRIDES["buzzwire4"]
SHORT_OVERRIDES["paper5"] = SHORT_OVERRIDES["buzzwire4"]
SHORT_OVERRIDES["buzzwire6"] = SHORT_OVERRIDES["buzzwire4"]
# A variant's colour and legend name are fixed globally, which breaks when the
# same run is the MAIN method in one tag's figure and an ablation in another's.
# ibmarl_strict_gated_a0.4 is buzzwire3_reg's headline method (red, "IBMARL")
# and buzzwire5's "w/o mixing" arm (teal).  Keyed by tag, applied by band() and
# fig_seeds; a tag or variant not listed keeps the module-level C / LABEL.
C_OVERRIDES = {
    "buzzwire3_reg": {"ibmarl_strict_gated_a0.4": "#E8253D",    # ours -> red
                      # ibmarl_strict is the ungated arm here, not the main
                      # method, so it cannot keep its global red.  Purple is
                      # what buzzwire5 already uses for "w/o gated term".
                      "ibmarl_strict": "#7B52AB"},
}
LABEL_OVERRIDES = {
    "buzzwire3_reg": {"ibmarl_strict_gated_a0.4": "IBMARL"},
}


def color_of(v):
    return C_OVERRIDES.get(TAG, {}).get(v, C[v])


def label_of(v):
    return LABEL_OVERRIDES.get(TAG, {}).get(v, LABEL[v])


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
# buzzwire3_reg is the alpha sweep of the gated term on top of buzzwire3's
# ibmarl_strict.  Its headline arm is ibmarl_strict_gated_a0.4; bc_eval,
# ibmarl_strict and the three baselines are symlinked in from buzzwire3 (which
# takes maddpg from buzzwire1 in turn), so every arm here shares one teacher
# (-137.5), one budget and one pre-2026-09-17 eval guard.
VARIANT_SETS["buzzwire3_reg"] = dict(
    baselines=["ibmarl_strict_gated_a0.4", "rlfd", "rft", "maddpg"],
    # ABLATIONS[0] == BASELINES[0] by convention (fig_seeds slices [1:]).
    # a0.1 and a1.6 have 2 seeds to a0.4's 5 -- the alpha sweep was a pilot.
    ablations=["ibmarl_strict_gated_a0.4", "ibmarl_strict_gated_a0.1",
               "ibmarl_strict_gated_a1.6", "ibmarl_strict"],
)
# transport800 runs the same eight arms under the same names (paper_run.py's
# 2026-09-20 registration), so it takes buzzwire4's variant set unchanged.
VARIANT_SETS["transport800"] = VARIANT_SETS["buzzwire4"]
# Poster tags: the same headline method and the two ablations the poster
# shows (no critic-ensemble arm was run under the legacy schema).
VARIANT_SETS["paper5"] = dict(
    baselines=["ibmarl_gated_a0.4", "rlfd", "rft", "maddpg"],
    ablations=["ibmarl_gated_a0.4", "ibmarl", "ibmarl_strict_gated_a0.4"],
)
VARIANT_SETS["buzzwire6"] = VARIANT_SETS["paper5"]
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
ABLATION_SHORT["buzzwire3_reg"] = {
    "ibmarl_strict_gated_a0.4": r"IBMARL ($\alpha$=0.4)",
    "ibmarl_strict_gated_a0.1": r"$\alpha$=0.1",
    "ibmarl_strict_gated_a1.6": r"$\alpha$=1.6",
    "ibmarl_strict":            "w/o gated term",
}
ABLATION_SHORT["transport800"] = ABLATION_SHORT["buzzwire4"]
ABLATION_SHORT["paper5"] = ABLATION_SHORT["buzzwire4"]
ABLATION_SHORT["buzzwire6"] = ABLATION_SHORT["buzzwire4"]

# Ablation figures that need more than one curve per variant.  fig_ablation()
# uses this when the tag has an entry, and falls back to one RL-actor curve per
# ABLATIONS entry otherwise, so every earlier tag's output is unchanged.
#
# buzzwire5 draws both evaluation protocols for the two variants that execute a
# mixing arbiter, because the gate's whole effect is on the gap between them:
# without the gated term the arbitrated policy is fine (0.89) while the RL
# actors it would have to be handed off to lag far behind (0.65); with it the
# two protocols nearly coincide.  A figure showing only one protocol cannot say
# that.  The strict and single-critic ablations get RL-actor curves only -- what
# they ablate is not about the hand-off, and six curves is already the most a
# \columnwidth panel carries.
#   variant, protocol ("rl" = RL actors alone, "executed" = arbitrated RL+IL),
#   legend label (None = no legend entry, the protocol legend explains it),
#   linestyle.
ABLATION_CURVES = {
    "buzzwire5": [
        ("ibmarl_gated_a0.4",         "rl",       "IBMARL",              "-"),
        ("ibmarl_gated_a0.4",         "executed", None,                  (0, (3, 1.6))),
        ("ibmarl",                    "rl",       "w/o gated term",      "-"),
        ("ibmarl",                    "executed", None,                  (0, (3, 1.6))),
        ("ibmarl_strict_gated_a0.4",  "rl",       "w/o mixing",          "-"),
        ("ibmarl_gated_a0.4_1critic", "rl",       "w/o critic ensemble", "-"),
    ],
}
# transport800: buzzwire3_reg's three-curve framing plus the mixing ablation.
# The ungated arm carries both protocols because its RL/RL+IL gap is what the
# term removes; the other two are RL-actor curves, so the panel reads as "what
# the decentralized actors reach" with one dashed reference for what the
# ungated method needs the teacher in the loop to reach.  Dropped from
# buzzwire5's six-curve set here: the gated arm's dashed twin and the
# single-critic arm, neither of which speaks to the hand-off.
ABLATION_CURVES["transport800"] = [
    ("ibmarl",                   "rl",       "w/o gated term", "-"),
    ("ibmarl",                   "executed", None,             (0, (3, 1.6))),
    ("ibmarl_gated_a0.4",        "rl",       "IBMARL",         "-"),
    ("ibmarl_strict_gated_a0.4", "rl",       "w/o mixing",     "-"),
]
# buzzwire3_reg: the gated term against the arm it was added to, three curves.
# The ungated arm gets both protocols because its RL/RL+IL gap is the thing the
# term removes; the gated arm is shown on the RL-actor protocol alone, and the
# figure's point is that it lands ABOVE the ungated arm's arbitrated policy
# (-71.2 vs -84.5) -- i.e. the decentralized actors beat what the ungated method
# can only reach with the teacher still in the loop.
ABLATION_CURVES["buzzwire3_reg"] = [
    ("ibmarl_strict",            "rl",       "w/o gated term",  "-"),
    ("ibmarl_strict",            "executed", None,              (0, (3, 1.6))),
    ("ibmarl_strict_gated_a0.4", "rl",       r"IBMARL (gated, $\alpha$=0.4)", "-"),
]
# Poster tags: buzzwire5's framing without the single-critic arm.
ABLATION_CURVES["paper5"] = [
    ("ibmarl_gated_a0.4",         "rl",       "IBMARL",              "-"),
    ("ibmarl_gated_a0.4",         "executed", None,                  (0, (3, 1.6))),
    ("ibmarl",                    "rl",       "w/o gated term",      "-"),
    ("ibmarl",                    "executed", None,                  (0, (3, 1.6))),
    ("ibmarl_strict_gated_a0.4",  "rl",       "w/o mixing",          "-"),
]
ABLATION_CURVES["buzzwire6"] = list(ABLATION_CURVES["paper5"])
# Same format as ABLATION_CURVES, for a figure that makes only the hand-off
# point: the two mixing-arbiter variants under both protocols, with the strict
# and single-critic ablations dropped.  fig2_ablation has to carry four
# variants at once, so the red/purple protocol gap -- the thing the gated term
# exists for -- competes with two curves that say nothing about it.  Here the
# panel holds four curves and the gap is the only comparison in it.
HANDOFF_CURVES = {
    "buzzwire5": [
        ("ibmarl_gated_a0.4", "rl",       "IBMARL (gated)",  "-"),
        ("ibmarl_gated_a0.4", "executed", None,              (0, (3, 1.6))),
        ("ibmarl",            "rl",       "w/o gated term",  "-"),
        ("ibmarl",            "executed", None,              (0, (3, 1.6))),
    ],
}
HANDOFF_CURVES["transport800"] = list(HANDOFF_CURVES["buzzwire5"])
HANDOFF_CURVES["paper5"] = list(HANDOFF_CURVES["buzzwire5"])
HANDOFF_CURVES["buzzwire6"] = list(HANDOFF_CURVES["buzzwire5"])
# Legend entries for the linestyle axis of a dual-protocol figure.
PROTOCOL_LEGEND = {
    "rl":       ("RL actors only", "-"),
    "executed": ("RL+IL (executed)", (0, (3, 1.6))),
}

RETURN_LABEL = "Episode return"


# --------------------------------------------------------------------------
def seed_dirs(variant, run=None, complete_only=False):
    """
    The variant's seed directories in results/<run>, seed-ordered.  With
    ``complete_only`` a seed whose status.json is missing or not ok is left
    out (a run still in flight or a crashed one); without it every seed with a
    metrics file counts, and curve() then truncates to the shortest.
    """
    import json
    run = run or TAG
    dirs = sorted(glob.glob(str(ROOT / "results" / run / variant / "seed_*")),
                  key=lambda p: int(p.rstrip("/").split("seed_")[-1]))
    dirs = [Path(d) for d in dirs if (Path(d) / "data" / "metrics.csv").exists()]
    if complete_only:
        keep = []
        for d in dirs:
            try:
                ok = json.load(open(d / "status.json")).get("ok")
            except (OSError, ValueError):
                ok = False
            if ok:
                keep.append(d)
            else:
                print(f"  .. {run}/{variant}/{d.name}: not finished, skipped")
        dirs = keep
    return dirs


def load(variant, run=None, complete_only=False):
    run = run or TAG
    frames = [pd.read_csv(d / "data" / "metrics.csv")
              for d in seed_dirs(variant, run, complete_only)]
    if not frames:
        raise FileNotFoundError(f"{run}/{variant}")
    return frames


def resolve_col(df, col, protocol=None):
    """
    Map the logical evaluation column onto the run's logged columns for the
    active PROTOCOL (or an explicit `protocol` override, used by figures that
    draw both protocols in one panel).  Only "eval_reward_mean" is
    protocol-dependent, and only for runs that logged eval_protocol == "rl_il"
    (IbmarlExperiment after the RL+IL evaluation was added); every other frame
    is returned unchanged.
    """
    if col != "eval_reward_mean" or "eval_protocol" not in df.columns:
        return col
    logged = df["eval_protocol"].dropna()
    if logged.empty or str(logged.iloc[0]) != "rl_il":
        return col
    return "rl_only_episode_reward_mean" if (protocol or PROTOCOL) == "rl" else col


def curve(variant, col, smooth=None, protocol=None, run=None, complete_only=False):
    """
    Steps (in thousands) and the per-seed matrix of the smoothed column.

    Seeds are truncated to the shortest one, so a still-running or crashed seed
    silently shortens every curve for that variant -- and, for steps-to-teacher,
    drags the "training budget" down with it.  Ragged lengths are therefore
    reported loudly rather than absorbed.  ``run`` reads another tag's
    results (cross-tag panels); ``complete_only`` drops unfinished seeds.
    """
    smooth = smooth or SMOOTH
    series, steps = [], None
    for d in load(variant, run, complete_only):
        c = resolve_col(d, col, protocol)
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
    c = color or color_of(variant)
    ax.plot(x, m, color=c, label=label_of(variant), zorder=3)
    ax.fill_between(x, m - sem, m + sem, color=c, alpha=0.18, lw=0, zorder=2)


def bc_values(run=None, variant="bc_eval"):
    """Per-seed teacher levels (first row of each bc_eval seed's metrics)."""
    root = ROOT / "results" / (run or TAG)
    vals = [pd.read_csv(f)["eval_reward_mean"].iloc[0]
            for f in sorted(glob.glob(str(root / variant / "seed_*" /
                                          "data" / "metrics.csv")))]
    if not vals:
        raise FileNotFoundError(f"{run or TAG}/{variant}")
    return [float(v) for v in vals]


def bc_level(run=None, variant="bc_eval"):
    return float(np.mean(bc_values(run, variant)))


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


def save(fig, name, protocol_suffix=True):
    OUT.mkdir(parents=True, exist_ok=True)
    # A non-default protocol gets its own files, so it never overwrites the
    # RL-actor-protocol figures the paper was built from.  SUFFIX does the same
    # for a partial figure (see --exclude): a figure missing a method must never
    # land on the filename of the complete one.
    # A figure that draws both protocols at once (protocol_suffix=False) is
    # identical whatever PROTOCOL is set to, so tagging it would produce two
    # filenames holding the same figure.
    if PROTOCOL != "rl" and protocol_suffix:
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
    # Anything absent -- whether it has no runs yet or was dropped by --exclude --
    # is stated on the figure itself.  The filename suffix is not enough: figures
    # get pasted into slides and issues without it.
    absent = missing + [v for v in EXCLUDED if v not in missing]
    if absent:
        ax.text(0.015, 0.03, "incomplete: no " + ", ".join(label_of(v) for v in absent),
                transform=ax.transAxes, fontsize=5.5,
                color="#B03030", va="bottom", ha="left", zorder=5)
    ax.axhline(bc_level(), color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
               label="R2BC (teacher)")
    style_return_axis(ax)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2, labelspacing=0.32)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    save(fig, "fig1_main")


# ============================== Figure 2 ==================================
def fig_ablation():
    """Ablation learning curves, single-column, matching fig1's layout."""
    if TAG in ABLATION_CURVES:
        return two_protocol_panel(ABLATION_CURVES[TAG], "fig2_ablation")
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
    ax.axhline(bc_level(), color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
               label="R2BC (teacher)")
    style_return_axis(ax)
    ax.legend(loc="lower right", ncol=1, borderpad=0.2, labelspacing=0.32)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    save(fig, "fig2_ablation")


def two_protocol_panel(curves, name, height=2.75):
    """
    One \columnwidth panel of learning curves with colour = variant and
    linestyle = evaluation protocol; `curves` is an ABLATION_CURVES-style list.
    Taller than fig1's 2.45 in because the legend carries a variant block and a
    protocol block.
    """
    fig, ax = plt.subplots(figsize=(COL_W, height))
    drawn_protocols, handles = [], []
    for variant, protocol, label, ls in curves:
        try:
            x, y = curve(variant, "eval_reward_mean", protocol=protocol)
        except FileNotFoundError:
            print(f"  .. {name}: skipping {variant} ({protocol}) "
                  f"(not in this sweep)")
            continue
        m, sem = mean_sem(y)
        col = color_of(variant)
        ax.plot(x, m, color=col, ls=ls, zorder=3, lw=1.15)
        # Overlapping bands turn the panel to mush, so the secondary protocol's
        # band is drawn fainter than the RL-actor one it qualifies.
        ax.fill_between(x, m - sem, m + sem, color=col,
                        alpha=0.15 if protocol == "rl" else 0.08,
                        lw=0, zorder=2)
        if label:
            # The variant's swatch is always drawn solid, so the colour block of
            # the legend reads as "which variant" and the protocol block below
            # is the only place linestyle means anything.
            handles.append(Line2D([], [], color=col, lw=1.6, label=label))
        if protocol not in drawn_protocols:
            drawn_protocols.append(protocol)
    bc = bc_level()
    ax.axhline(bc, color=C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1)
    handles.append(Line2D([], [], color=C["bc"], ls=(0, (4, 2.5)), lw=1.1,
                          label="R2BC (teacher)"))
    if len(drawn_protocols) > 1:
        handles.append(Line2D([], [], ls="none", label=" "))   # block separator
        for pr in drawn_protocols:
            pr_label, pr_ls = PROTOCOL_LEGEND[pr]
            handles.append(Line2D([], [], color="0.35", ls=pr_ls, lw=1.15,
                                  label=pr_label))
    style_return_axis(ax)
    # Which corner is empty depends on the task's curve shape, so it is a
    # per-tag preset (see PRESETS[...]["legend_loc"]) rather than a constant.
    ax.legend(handles=handles, loc=PRESET.get("legend_loc", "lower right"),
              ncol=PRESET.get("legend_ncol", 1), fontsize=6.6,
              borderpad=0.2, labelspacing=0.28, handlelength=1.9,
              borderaxespad=0.35)
    fig.subplots_adjust(left=0.155, right=0.968, top=0.975, bottom=0.165)
    # Both protocols are on the one figure, so it must not take PROTOCOL's
    # filename suffix -- that would write the same picture under two names.
    save(fig, name, protocol_suffix=False)


def fig_handoff():
    """The gated term's effect on the RL-actor / RL+IL gap (HANDOFF_CURVES)."""
    two_protocol_panel(HANDOFF_CURVES[TAG], "fig5_handoff_gap")


# ============================== Figure 3 ==================================
def fig_seeds():
    """Per-seed final return, i.e. run-to-run reliability."""
    bc = bc_level()
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
                   finals, s=11, color=color_of(v), zorder=3, linewidths=0)
        ax.hlines(finals.mean(), i - 0.28, i + 0.28, color=color_of(v), lw=1.6,
                  zorder=2)
    ax.axhline(bc, color=C["bc"], ls=(0, (4, 2.5)), lw=1.0, zorder=1)
    ax.set_xlim(-0.6, len(order) - 0.4)
    # Offset sized per-preset: 1.2 (the historical default) reads fine against
    # the legacy return scale (tens of units) but would push the label off a
    # [0, 1]-ranged success-rate axis invisibly, with no error to flag it.
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
    if TAG in HANDOFF_CURVES:
        fig_handoff()
    if TAG == "paper3":
        # Nav-specific pre-fix/post-fix comparison; needs both sweeps'
        # checkpoint_eval_fixed.csv.
        fig_replication()
