"""
Before/after comparison for the gated imitation term (config ``actor_reg``).

    python -m analysis.actor_reg_comparison --tag buzzwire3_reg \
        --before ibmarl_strict --after ibmarl_strict_gated_a0.4

The question the figure answers: does the RL actor alone (the decentralised
policy, ``rl_only_episode_reward_mean``) catch up with the arbitrated RL+IL
policy (``eval_reward_mean``, needs the centralised critics) once the actor
loss carries the gated imitation term?  Both protocols are logged by every
IBMARL run on the same eval-env seeds, so the four curves are paired.

Left panel: mean +/- s.e.m. over seeds of the two protocols for the before arm
and each after arm (solid = RL+IL, dashed = RL actor), plus the teacher level.
Right panel: the after arms' mean gate (the arbiter's preference for the
teacher on training minibatches) and the RL-action fraction of the arbitrated
evaluation, both in [0, 1].  A gate that starts high and decays is the
hypothesis working as intended; one that never falls means the critic keeps
preferring the teacher.

Prints steps-to-teacher (steps_to_teacher.py's definition: first step where
the SMOOTH-evaluation centred rolling mean exceeds the teacher level, median
over seeds, non-reaching seeds capped at the budget and counted) and the
last-10-evaluation final return (mean +/- s.d. over seeds) for every curve.

Reuses paper3_figures' load/curve/mean_sem/bc_level (with the smoothing of
the paper figures) and fig_mixing_comparison's load_complete so an unfinished
seed of an in-flight sweep is dropped rather than truncating every curve.
Writes analysis/figures/<tag>/fig_actor_reg.{pdf,png}.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import paper3_figures as pf
from analysis.fig_mixing_comparison import load_complete

# Logical protocol name -> logged column.  Read literally: configure() is
# called with protocol="executed" so resolve_col leaves "eval_reward_mean" as
# the arbitrated RL+IL column, and the RL-only column is never remapped.
PROTOCOLS = {"RL+IL": "eval_reward_mean", "RL actor": "rl_only_episode_reward_mean"}
STYLE = {"RL+IL": "-", "RL actor": "--"}

AFTER_COLORS = ["#1B9E77", "#66C2A5", "#0B5D47", "#7570B3"]


def crossings(x, y, teacher):
    """Per-seed first step (in thousands) the smoothed curve exceeds *teacher*; NaN if never."""
    out = []
    for row in y:
        hit = np.nonzero(row > teacher)[0]
        out.append(x[hit[0]] if len(hit) else np.nan)
    return np.array(out, dtype=float)


def final_return(variant, col):
    """Per-seed mean of the last 10 raw evaluations of *col*."""
    vals = []
    for df in pf.load(variant):
        sub = df[df["group"] == df["group"].iloc[0]][col].dropna()
        if len(sub):
            vals.append(sub.tail(10).mean())
    return np.array(vals, dtype=float)


def summarise(variant, proto, teacher, budget):
    col = PROTOCOLS[proto]
    x, y = pf.curve(variant, col)
    cross = crossings(x, y, teacher)
    reached = int(np.isfinite(cross).sum())
    capped = np.where(np.isfinite(cross), cross, budget)
    fin = final_return(variant, col)
    return dict(
        variant=variant, protocol=proto, n=y.shape[0],
        steps_k=float(np.median(capped)), reached=reached,
        steps_k_reaching=float(np.median(cross[np.isfinite(cross)])) if reached else np.nan,
        final_mean=float(fin.mean()), final_sd=float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="buzzwire3_reg")
    ap.add_argument("--before", default="ibmarl_strict")
    ap.add_argument("--after", nargs="+", default=["ibmarl_strict_gated_a0.4"])
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--name", default="fig_actor_reg", help="output file stem")
    ap.add_argument("--returns-only", action="store_true",
                    help="only the return panel (no gate / arbiter-choice panel)")
    args = ap.parse_args()

    pf.configure(args.tag, args.smooth, protocol="executed")
    pf.load = load_complete  # drop unfinished seeds instead of truncating everything
    teacher = pf.bc_level()

    # --- figure -------------------------------------------------------------
    if args.returns_only:
        fig, ax = plt.subplots(figsize=(1.3 * pf.COL_W, 2.8))
        ax2 = None
    else:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(2 * pf.COL_W, 2.6),
                                      gridspec_kw=dict(width_ratios=[1.35, 1]))
    arms = [(args.before, pf.C.get(args.before, "#E8253D"))] + [
        (v, pf.C.get(v, AFTER_COLORS[i % len(AFTER_COLORS)]))
        for i, v in enumerate(args.after)
    ]
    rows, budget = [], None
    for variant, color in arms:
        for proto, col in PROTOCOLS.items():
            x, y = pf.curve(variant, col)
            budget = budget or float(x[-1])
            m, sem = pf.mean_sem(y)
            label = f"{pf.LABEL.get(variant, variant)}, {proto} (n={y.shape[0]})"
            ax.plot(x, m, STYLE[proto], color=color, label=label, zorder=3, lw=1.1)
            ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.14, lw=0, zorder=2)
            rows.append(summarise(variant, proto, teacher, budget))
    ax.axhline(teacher, color=pf.C["bc"], ls=":", lw=1.0, label="IL policy (R2BC)", zorder=1)
    pf.style_return_axis(ax, title="Executed (RL+IL) vs RL actor alone")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)

    # Right panel: gate and eval RL-action fraction.
    for variant, color in (arms if ax2 is not None else []):
        try:
            x, y = pf.curve(variant, "eval_rl_action_fraction")
            m, sem = pf.mean_sem(y)
            ax2.plot(x, m, "-", color=color, lw=1.0,
                     label=f"{pf.LABEL.get(variant, variant)}: eval RL-action fraction")
            ax2.fill_between(x, m - sem, m + sem, color=color, alpha=0.14, lw=0)
        except (FileNotFoundError, KeyError):
            pass
        if variant == args.before:
            continue
        try:
            x, y = pf.curve(variant, "gate_mean")
        except KeyError:
            print(f"  .. {variant}: no gate_mean column (actor_reg off?)")
            continue
        m, sem = pf.mean_sem(y)
        ax2.plot(x, m, "-.", color=color, lw=1.2,
                 label=f"{pf.LABEL.get(variant, variant)}: mean gate")
        ax2.fill_between(x, m - sem, m + sem, color=color, alpha=0.14, lw=0)
    if ax2 is not None:
        ax2.set_ylim(0, 1)
        ax2.set_xlim(*pf.PRESET["xlim"]) if pf.PRESET["xlim"] else None
        ax2.set_xlabel(r"Interaction steps ($\times$1000)")
        ax2.set_ylabel("Fraction")
        ax2.set_title("Gate and arbiter choice", pad=4)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(loc="best", frameon=False, fontsize=6)

    fig.tight_layout()
    pf.OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(pf.OUT / f"{args.name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print(f"wrote {pf.OUT / args.name}.pdf / .png")

    # --- table --------------------------------------------------------------
    print(f"\ntag={args.tag}  teacher level = {teacher:.1f}  smooth={pf.SMOOTH} evals  "
          f"budget={budget:.0f}k\n")
    hdr = (f"{'variant':<28} {'protocol':<9} {'n':>2} {'steps-to-teacher':>17} "
           f"{'reached':>8} {'median(reaching)':>17} {'final (last 10)':>17}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flag = "" if r["reached"] == r["n"] else "†"
        reaching = f"{r['steps_k_reaching']:.0f}k" if r["reached"] else "--"
        print(f"{r['variant']:<28} {r['protocol']:<9} {r['n']:>2} "
              f"{r['steps_k']:>15.0f}k{flag:<1} {r['reached']:>3}/{r['n']:<4} "
              f"{reaching:>17} {r['final_mean']:>9.1f} ± {r['final_sd']:<5.1f}")
    print("\n† = capped at the training budget (seed(s) never reached the teacher)")
    out_csv = pf.OUT / f"{args.name}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
