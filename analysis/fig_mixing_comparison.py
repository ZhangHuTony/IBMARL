"""
Four-way comparison: MADDPG, IBMARL (joint choice), IBMARL w/ per-agent mixing,
RLfD -- learning curves plus a summary table.

    python -m analysis.fig_mixing_comparison --tag buzzwire1

This is the arbiter-design comparison: `ibmarl_strict` picks all-IL or all-RL as
one joint action, while `ibmarl` scores all 2^N per-agent combinations
(paper_run.VARIANTS).  On a task with heterogeneous agent roles like buzz_wire
those can come apart, which the navigation sweep had no way to show.

RFT is deliberately absent -- it is still running when this is generated.  Style,
smoothing and the teacher line are reused verbatim from analysis.paper3_figures
so the output drops in beside fig1/fig2 without a second visual language.

Writes analysis/figures/<tag>/fig_mixing_comparison.{pdf,png}.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import paper3_figures as pf

# Ordered worst-to-best so the legend reads bottom-up like the curves do.
VARIANTS = ["maddpg", "rlfd", "ibmarl", "ibmarl_strict"]

N_SEEDS: dict[str, int] = {}

# When True, load_complete additionally drops seeds that never reached the
# teacher.  This is a SELECTION-BIASED view -- see fig_mixing() for the caveat.
REACHING_ONLY = False


def _reached(df, teacher: float) -> bool:
    """Table-I criterion: smoothed eval curve exceeds *teacher* at any point."""
    sub = df[df["group"] == df["group"].iloc[0]][pf.resolve_col(df, "eval_reward_mean")].dropna()
    sm = sub.rolling(pf.SMOOTH, center=True, min_periods=1).mean()
    return bool((sm > teacher).any())


def load_complete(variant, run=None):
    """
    Like paper3_figures.load, but drops seeds whose status.json is missing or
    not ok.

    Necessary while the sweep is in flight: curve() truncates every seed to the
    shortest one, so a single half-finished seed would silently collapse the
    whole figure to a handful of evaluations.  Dropping it costs one seed of
    statistics; keeping it would corrupt every curve in the panel.
    """
    run = run or pf.TAG
    frames, kept = [], 0
    pattern = str(pf.ROOT / "results" / run / variant / "seed_*")
    for d in sorted(glob.glob(pattern),
                    key=lambda p: int(p.split("seed_")[1])):
        status = Path(d) / "status.json"
        try:
            ok = json.load(open(status)).get("ok")
        except (json.JSONDecodeError, OSError):
            ok = False
        if not ok:
            print(f"  .. {variant}/{Path(d).name}: unfinished, excluded")
            continue
        df = pd.read_csv(Path(d) / "data" / "metrics.csv")
        if REACHING_ONLY and not _reached(df, pf.bc_level()):
            print(f"  .. {variant}/{Path(d).name}: never reached teacher, excluded")
            continue
        frames.append(df)
        kept += 1
    if not frames:
        raise FileNotFoundError(f"{run}/{variant}: no completed seeds")
    N_SEEDS[variant] = kept
    return frames


def summary_table(teacher: float) -> None:
    """Steps-to-teacher and final return, same definitions as Table I."""
    hdr = (f"{'variant':<28} {'reached':>8} {'median steps (reaching)':>24} "
           f"{'final return':>16}")
    print(f"\ntag={pf.TAG}  teacher = {teacher:.1f}  (smooth={pf.SMOOTH}, "
          f"protocol={pf.PROTOCOL})\n")
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        try:
            frames = pf.load(v)
        except FileNotFoundError:
            print(f"{pf.LABEL[v]:<28} {'-- not run --':>8}")
            continue
        steps_k, smoothed = pf.curve(v, "eval_reward_mean")
        steps = steps_k * 1000.0
        per_seed, reached = [], []
        for y in smoothed:
            idx = np.flatnonzero(y > teacher)
            per_seed.append(steps[idx[0]] if len(idx) else steps[-1])
            reached.append(bool(len(idx)))
        per_seed, reached = np.array(per_seed), np.array(reached)
        finals = np.array([d[pf.resolve_col(d, "eval_reward_mean")].dropna().tail(10).mean()
                           for d in frames])
        med = (f"{np.median(per_seed[reached])/1000:>23.0f}k"
               if reached.any() else f"{'--':>24}")
        sd = finals.std(ddof=1) if len(finals) > 1 else 0.0
        print(f"{pf.LABEL[v]:<28} {reached.sum():>4}/{len(per_seed):<3} {med}  "
              f"{finals.mean():>8.1f} +/- {sd:.1f}")
    ns = {v: N_SEEDS.get(v) for v in VARIANTS if v in N_SEEDS}
    print(f"\nseeds per variant: {ns}")


def fig_mixing() -> None:
    """
    Four-way comparison panel.

    With REACHING_ONLY the panel shows only seeds that reached the teacher.  That
    is a conditional-on-success view: it answers "when a method works, how well
    and how fast?", NOT "how well does this method do?".  It cannot be read as a
    method comparison, because the variants are conditioned on different numbers
    of seeds (and MADDPG vanishes entirely at 0/5).  Always report it beside the
    unconditional panel and the n's, never on its own.
    """
    fig, ax = pf.plt.subplots(figsize=(pf.COL_W, 2.45))
    for v in VARIANTS:
        try:
            pf.band(ax, v, "eval_reward_mean")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  .. skipping {v}: {exc}")
    teacher = pf.bc_level()
    ax.axhline(teacher, color=pf.C["bc"], ls=(0, (4, 2.5)), lw=1.1, zorder=1,
               label="R2BC (teacher)")
    pf.style_return_axis(ax)
    # Re-label with the surviving seed count so the panel cannot be read as if
    # every variant contributed five runs.
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{l} (n={N_SEEDS[v]})" if v in N_SEEDS and pf.LABEL.get(v) == l
              else l for l, v in zip(labels, VARIANTS + ["bc"])]
    ax.legend(handles, labels, loc="lower right", ncol=1, borderpad=0.2,
              labelspacing=0.32, fontsize=6.4)
    fig.subplots_adjust(left=0.175, right=0.968, top=0.975, bottom=0.185)
    pf.save(fig, "fig_mixing_comparison_successes" if REACHING_ONLY
                 else "fig_mixing_comparison")
    summary_table(teacher)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="buzzwire1")
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--successes-only", action="store_true",
                    help="Plot only seeds that reached the teacher. "
                         "Selection-biased; see fig_mixing.__doc__.")
    ap.add_argument("--protocol", choices=["rl", "executed"], default="rl",
                    help="rl: RL-actor evaluation for every run (default); "
                         "executed: each method's executed policy (IBMARL's "
                         "arbitrated RL+IL where logged)")
    args = ap.parse_args()
    pf.configure(args.tag, args.smooth, args.protocol)
    # Exclude unfinished seeds; see load_complete's docstring for why a single
    # in-flight seed would otherwise truncate every curve in the panel.
    pf.load = load_complete
    global REACHING_ONLY
    REACHING_ONLY = args.successes_only
    fig_mixing()


if __name__ == "__main__":
    main()
