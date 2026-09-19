"""
Table-I statistics for a paper sweep: steps-to-teacher and final return.

    python -m analysis.steps_to_teacher                 # paper3 (navigation)
    python -m analysis.steps_to_teacher --tag buzzwire1

Definitions (CONTEXT.md / the workshop draft's Table I):

  steps to teacher  Environment steps until the SMOOTH-evaluation centred
                    rolling mean of eval_reward_mean first exceeds the teacher
                    level (bc_eval first-row mean across seeds).  Reported as
                    the median over all seeds, with seeds that never reach the
                    teacher capped at the training budget and flagged; the
                    median over reaching seeds only is also printed.
  final return      Mean of the last 10 evaluations per seed, then
                    mean +/- s.d. over seeds.

Writes analysis/figures/<tag>/steps_to_teacher.csv and prints the table.
Reuses load()/curve()/bc_level()/SMOOTH from analysis.paper3_figures so the
smoothing is bit-identical to the figures.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from analysis import paper3_figures as pf


def unfinished_seeds(tag: str, variant: str) -> list[str]:
    """Seed dirs whose status.json is missing or not ok (still running/failed)."""
    bad = []
    for d in sorted(glob.glob(str(pf.ROOT / "results" / tag / variant / "seed_*"))):
        status = Path(d) / "status.json"
        try:
            if json.load(open(status)).get("ok"):
                continue
        except (json.JSONDecodeError, OSError):
            pass
        bad.append(Path(d).name)
    return bad

ORDER = [
    "ibmarl_strict",
    "rlfd",
    "rft",
    "maddpg",
    "ibmarl",
    "ibmarl_strict_hard",
    "ibmarl_strict_1critic",
    # actor-lag experiment (results/buzzwire3_reg): silently skipped elsewhere
    "ibmarl_strict_gated_a0.1",
    "ibmarl_strict_gated_a0.4",
    "ibmarl_strict_gated_a1.6",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="paper3")
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--complete-only", action="store_true",
                    help="Skip variants with any unfinished seed (status.json "
                         "missing or not ok). Use while a sweep is in flight.")
    ap.add_argument("--protocol", choices=["rl", "executed"], default="rl",
                    help="rl: RL-actor evaluation for every run (default, the "
                         "steps-to-teacher protocol); executed: each method's "
                         "executed policy (IBMARL's arbitrated RL+IL where logged)")
    args = ap.parse_args()
    pf.configure(args.tag, args.smooth, args.protocol)

    teacher = pf.bc_level()
    rows = []
    print(f"\ntag={args.tag}  teacher level = {teacher:.1f}  "
          f"(smooth={pf.SMOOTH} evals, protocol={pf.PROTOCOL})\n")
    hdr = (f"{'variant':<24} {'steps-to-teacher':>18} {'reached':>8} "
           f"{'median(reaching)':>17} {'final return':>16}")
    print(hdr)
    print("-" * len(hdr))

    for variant in ORDER:
        try:
            frames = pf.load(variant)
        except FileNotFoundError:
            continue
        unfinished = unfinished_seeds(args.tag, variant)
        if unfinished:
            msg = (f"{variant}: {len(unfinished)} seed(s) not finished "
                   f"({', '.join(unfinished)})")
            if args.complete_only:
                print(f"  .. skipping {msg}")
                continue
            print(f"  !! {msg} -- numbers below are provisional")
        steps_k, smoothed = pf.curve(variant, "eval_reward_mean")
        steps = steps_k * 1000.0
        budget = steps[-1]

        per_seed, reached = [], []
        for y in smoothed:
            idx = np.flatnonzero(y > teacher)
            if len(idx):
                per_seed.append(steps[idx[0]])
                reached.append(True)
            else:
                per_seed.append(budget)
                reached.append(False)
        per_seed = np.array(per_seed)
        reached = np.array(reached)

        med_all = float(np.median(per_seed))
        med_reach = (float(np.median(per_seed[reached]))
                     if reached.any() else float("nan"))
        finals = np.array([d[pf.resolve_col(d, "eval_reward_mean")].dropna().tail(10).mean()
                           for d in frames])
        capped = not reached.all()

        rows.append(dict(
            variant=variant,
            n_seeds=len(per_seed),
            n_reached=int(reached.sum()),
            steps_to_teacher_median=med_all,
            steps_to_teacher_median_reaching=med_reach,
            capped_at_budget=capped,
            final_return_mean=float(finals.mean()),
            final_return_sd=float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
        ))
        flag = "†" if capped else " "
        med_reach_s = ("--".rjust(16) if np.isnan(med_reach)
                       else f"{med_reach/1000:>15.0f}k")
        print(f"{pf.LABEL[variant]:<24} {med_all/1000:>15.0f}k{flag} "
              f"{reached.sum():>4}/{len(per_seed):<3} "
              f"{med_reach_s}  "
              f"{finals.mean():>8.1f} ± {rows[-1]['final_return_sd']:.1f}")

    import pandas as pd
    out = pf.OUT / ("steps_to_teacher.csv" if pf.PROTOCOL == "rl"
                    else f"steps_to_teacher_{pf.PROTOCOL}.csv")
    pf.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n† = capped at the training budget (seed(s) never reached "
          f"the teacher)\nwrote {out}")


if __name__ == "__main__":
    main()
