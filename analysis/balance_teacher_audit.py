"""
How strong is the balance teacher, and at which basin radius?

Under the gt-in-radius schema (config/environments/balance.yaml) the radius is
what makes the task sparse, and it also sets how strong the frozen teacher
looks: every evaluation statistic of the sweep is read against the teacher
level, so the radius has to be pinned BEFORE the arms run.  The buzz-wire
precedent targets a teacher that succeeds in roughly half its episodes (0.49,
config/experiments/ibmarl_buzz_wire.yaml); this prints the equivalent table.

One set of deterministic teacher rollouts on the dedicated eval env serves every
radius: the env is built with ``gt_radius = inf`` so BalanceSparseReward is the
identity and the native reward is logged, while ``done()`` is still suppressed
(FixedHorizonBalanceScenario), so a rollout of ``horizon`` steps is exactly one
episode per sub-env.  Per radius the table reports

  reach      fraction of episodes whose package ever came within the radius,
             with a binomial SEM over episodes (the "success rate")
  first      25/50/75th percentile of the first-entry step, over reaching episodes
  inside     mean steps spent inside the basin
  return     mean of sum_t [native_t if inside_t else -1] -- what bc_eval would log

plus radius-independent numbers: quantiles of the per-episode minimum distance,
the initial distance, floor-contact (fall) rate and the mean native return
(comparable to R2BC's own ~-220 note for this checkpoint).  With --demos the
same table is printed for the recording the teacher was trained on
(analysis/relabel_demos_sparse.py), so the two can be read side by side.

    python -m analysis.balance_teacher_audit --episodes 400 --radii 0.3,0.5,0.75,1.05 \
        teacher=teachers/balance/policy_checkpoint.pth --demos teachers/balance/demonstrations.pt
    python -m analysis.balance_teacher_audit k18=teachers/balance_18demo/policy_checkpoint.pth ...

Not a mode of teacher_success_curve.py: that one asserts a 0/1 team return and
puts a binomial SEM on the return itself, which only the binary schema admits.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
from pathlib import Path

import numpy as np
import torch
from torchrl.envs import ExplorationType, set_exploration_type

from analysis.paper_run import build_cfg
from analysis.relabel_demos_sparse import TASKS as DEMO_TASKS, episode_view, load_recording, print_table, radius_stats, shape_stats
from src.experiments.bc_eval_experiment import BcEvalExperiment

REL = slice(8, 10)
OUT_PENALTY = -1.0
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def teacher_rollouts(exp: BcEvalExperiment, episodes: int, seed: int):
    """Returns (dist [k, T], native [k, T], dist0 [k]) over k >= episodes episodes."""
    env = exp.eval_env
    n_envs = int(env.batch_size[0])
    horizon = int(exp.config["horizon"])
    exp.reseed_eval_env(seed)
    dists, natives, dist0s = [], [], []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(max(1, -(-episodes // n_envs))):
            out = env.rollout(horizon, policy=exp._bc_td_policy, break_when_any_done=False)
            done = out.get(("next", "done")).reshape(n_envs, horizon)
            assert not done[:, :-1].any() and done[:, -1].all(), \
                "episode ended before the horizon: done() is not suppressed (suppress_done False?)"
            nxt = out.get(("next", "agents", "observation"))            # [n_envs, T, A, 16]
            dists.append(torch.linalg.vector_norm(nxt[..., 0, REL], dim=-1))
            natives.append(out.get(("next", "agents", "reward"))[..., 0, 0])
            first = out.get(("agents", "observation"))[:, 0, 0, REL]
            dist0s.append(torch.linalg.vector_norm(first, dim=-1))
    cat = lambda xs: torch.cat(xs, 0).cpu().numpy()
    return cat(dists), cat(natives), cat(dist0s)


def policy_table(dist: np.ndarray, native: np.ndarray, radii: list[float]) -> list[dict]:
    rows = []
    for r in radii:
        inside = dist < r
        reach = inside.any(1)
        k = reach.size
        p = float(reach.mean())
        first = inside.argmax(1)[reach] + 1
        ret = np.where(inside, native, OUT_PENALTY).sum(1)
        rows.append(dict(
            radius=float(r),
            episodes_ever_inside=f"{int(reach.sum())}/{k}",
            reach=round(p, 3), reach_sem=round(math.sqrt(p * (1 - p) / k), 3),
            frac_rows_inside=round(float(inside.mean()), 4),
            first_entry_step_q25_50_75=[int(x) for x in np.percentile(first, [25, 50, 75])] if reach.any() else None,
            steps_inside_mean=round(float(inside.sum(1).mean()), 1),
            return_mean=round(float(ret.mean()), 1),
            return_min=round(float(ret.min()), 1),
            return_max=round(float(ret.max()), 1),
        ))
    return rows


def policy_shape(dist: np.ndarray, native: np.ndarray, dist0: np.ndarray) -> dict:
    fall = native < -5.0
    return dict(
        n_episodes=int(dist.shape[0]),
        initial_dist_min_max=[round(float(dist0.min()), 3), round(float(dist0.max()), 3)],
        min_dist_quantiles={str(q): round(float(np.quantile(dist.min(1), q)), 3) for q in QUANTILES},
        final_dist_mean=round(float(dist[:, -1].mean()), 3),
        native_return_mean=round(float(native.sum(1).mean()), 1),
        fall_rows=int(fall.sum()),
        episodes_with_fall=int(fall.any(1).sum()),
    )


def in_band(rows: list[dict], lo: float, hi: float) -> dict | None:
    """Smallest radius whose reach lies in [lo, hi]."""
    for r in sorted(rows, key=lambda x: x["radius"]):
        if lo <= r["reach"] <= hi:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", nargs="+", help="name=path/to/policy_checkpoint.pth")
    ap.add_argument("--scenario", default="balance")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--radii", default="0.3,0.5,0.75,1.05")
    ap.add_argument("--seed", type=int, default=0, help="eval-env seed, shared by every candidate")
    ap.add_argument("--eval-batch", type=int, default=100, help="sub-envs of the eval env (episodes per rollout)")
    ap.add_argument("--band", default="0.4,0.6", help="target reach band for the recommendation")
    ap.add_argument("--demos", type=Path, default=None, help="recording to tabulate beside the policy")
    ap.add_argument("--csv", type=Path, default=None, help="append the per-radius rows here")
    args = ap.parse_args()

    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    lo, hi = (float(x) for x in args.band.split(","))
    all_rows = []

    if args.demos is not None:
        spec = DEMO_TASKS[args.scenario]
        rec = load_recording(args.demos)
        ep, n_envs, cont = episode_view(rec, spec["horizon"])
        print(f"\n=== demonstrations: {args.demos} (env interleave {n_envs}, continuity {cont:.4f}) ===")
        for k, v in shape_stats(ep, spec).items():
            print(f"  {k}: {v}")
        rows = [radius_stats(ep, spec, r) for r in radii]
        for r in rows:
            r["reach"] = round(r["episodes_ever_inside"] / ep["obs"].shape[0], 3)
        print_table(rows)

    for spec_str in args.candidates:
        name, path = spec_str.split("=", 1)
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = build_cfg("bc_eval", args.scenario, 0, None, tag="_teacher_audit")
            cfg.update(r2bc_checkpoint_path=path, sparse_rewards=True, suppress_done=True,
                       gt_radius=float("inf"), eval_episodes=args.eval_batch)
            exp = BcEvalExperiment(cfg)
            dist, native, dist0 = teacher_rollouts(exp, args.episodes, args.seed)
        print(f"\n=== teacher {name}: {path}  ({dist.shape[0]} episodes, horizon {dist.shape[1]}, seed {args.seed}) ===")
        for k, v in policy_shape(dist, native, dist0).items():
            print(f"  {k}: {v}")
        rows = policy_table(dist, native, radii)
        cols = ["radius", "episodes_ever_inside", "reach", "reach_sem", "frac_rows_inside",
                "first_entry_step_q25_50_75", "steps_inside_mean", "return_mean", "return_min", "return_max"]
        print("  ".join(f"{c:>26}" for c in cols))
        for r in rows:
            print("  ".join(f"{str(r[c]):>26}" for c in cols))
        pick = in_band(rows, lo, hi)
        if pick is None:
            print(f"  -> no radius in the grid puts reach in [{lo}, {hi}]: "
                  f"{'teacher too strong at every radius' if min(r['reach'] for r in rows) > hi else 'teacher too weak at every radius'}")
        else:
            print(f"  -> smallest radius in [{lo}, {hi}]: {pick['radius']:g} (reach {pick['reach']:.3f} +/- {pick['reach_sem']:.3f})")
        for r in rows:
            all_rows.append(dict(teacher=name, checkpoint=path, episodes=dist.shape[0], seed=args.seed, **r))

    if args.csv is not None and all_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.exists()
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(all_rows)
        print(f"\nappended {len(all_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
