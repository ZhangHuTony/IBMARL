"""
Post-hoc evaluation of saved policy checkpoints.

Every run under ``results/<tag>/`` stores its final policy in
``checkpoints/policy_checkpoint.pt``.  This script reloads those weights and
re-measures them with the *current* ``BaseMARLExperiment.evaluate()``, in a
clean environment, without retraining anything.

Why this exists
---------------
Two independent bugs affected the runs in ``results/paper/``:

  A. ``evaluate()`` read a key that only exists on collector batches, so the
     logged ``eval_reward_mean`` was a time-average of a partial accumulator --
     roughly half the true episode return.  A *measurement* bug.
  B. ``evaluate()`` rolled out the collector's own (stateful) VMAS env, so ~1%
     of collected transitions paired a stale observation with an action and
     next-state from a different world state before entering the replay
     buffer.  A *training-data* bug, baked into the weights.

Comparing the old numbers against a fresh sweep confounds the two.  This script
supplies the missing middle term:

    (a) results/ as logged      old policy, old (broken) metric
    (b) THIS SCRIPT             old policy, fixed metric
    (c) fresh sweep             new policy, fixed metric

``b - a`` isolates the metric bug; ``c - b`` isolates the training bug.

It doubles as a smoke-test of the fixed ``evaluate()`` against real trained
weights across every variant -- worth knowing before committing ~9 hours of
wall clock to a new sweep.

Note this does NOT rehabilitate the old runs: those policies were still trained
on corrupted data.  It measures what they actually achieve, nothing more.

Usage
-----
    python -m analysis.eval_checkpoints                      # results/paper
    python -m analysis.eval_checkpoints --episodes 500
    python -m analysis.eval_checkpoints --only ibmarl,maddpg
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from statistics import mean, stdev

import torch
import yaml

from src.experiments.base_marl_experiment import BaseMARLExperiment


class _CheckpointEvaluator(BaseMARLExperiment):
    """
    Minimal concrete experiment whose only job is to expose ``evaluate()``.

    Deliberately does not reimplement the evaluation loop: reusing the real
    method is what makes this a valid check of the fix.
    """

    def train(self) -> str:  # pragma: no cover - never called
        raise NotImplementedError("eval-only harness")


def _build_policies(exp_type: str, cfg: dict, env, device) -> dict:
    """Reconstruct the architecture a run's checkpoint was saved from."""
    if exp_type == "ibmarl":
        from src.experiments.ibmarl.networks import build_rl_policies

        return build_rl_policies(cfg, env, device)

    if exp_type in ("maddpg", "rlfd", "rft"):
        from src.experiments.maddpg import MaddpgExperiment

        # _setup_policy never touches self; calling it unbound avoids building
        # the collectors, replay buffers and demo loaders a full experiment
        # would drag in.
        policies, _exploration = MaddpgExperiment._setup_policy(
            None, cfg, env, device
        )
        return policies

    raise ValueError(f"no policy builder for exp_type={exp_type!r}")


def _last_logged_eval(run_dir: Path) -> dict:
    """Final logged ``eval_reward_mean`` per group, i.e. column (a)."""
    metrics = run_dir / "data" / "metrics.csv"
    if not metrics.exists():
        return {}
    latest: dict[str, float] = {}
    with metrics.open() as f:
        for row in csv.DictReader(f):
            raw = (row.get("eval_reward_mean") or "").strip()
            if raw:
                try:
                    latest[row["group"]] = float(raw)
                except ValueError:
                    pass
    return latest


def _evaluate_run(run_dir: Path, n_episodes: int, scratch: Path) -> dict | None:
    """Re-measure one run's final checkpoint.  Returns None if not evaluable."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    exp_type = cfg.get("exp_type")

    ckpt_path = run_dir / "checkpoints" / "policy_checkpoint.pt"
    if exp_type != "bc_eval" and not ckpt_path.exists():
        print(f"  skip {run_dir}: no policy_checkpoint.pt")
        return None

    # Redirect every write away from the run directory.  Nothing under
    # results/ is opened for writing by this script.
    sandbox = scratch / run_dir.parent.name / run_dir.name
    for key in ("data_dir", "videos_dir", "check_dir", "plots_dir"):
        cfg[key] = str(sandbox / key)
    (sandbox / "data_dir").mkdir(parents=True, exist_ok=True)

    exp = _CheckpointEvaluator(cfg)

    if exp_type == "bc_eval":
        # No checkpoint of its own: bc_eval evaluates the frozen R2BC policy.
        from src.experiments.bc_eval_experiment import BcEvalExperiment

        bc = BcEvalExperiment(cfg)
        exp.policies = {"bc": bc._bc_td_policy}
    else:
        policies = _build_policies(exp_type, cfg, exp.env, exp.device)
        state = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
        missing = set(policies) - set(state)
        if missing:
            raise RuntimeError(f"{run_dir}: checkpoint missing groups {missing}")
        for group, policy in policies.items():
            policy.load_state_dict(state[group])
        exp.policies = policies

    fresh = exp.evaluate(n_episodes=n_episodes)
    logged = _last_logged_eval(run_dir)

    return {
        "variant": run_dir.parent.name,
        "seed": cfg.get("seed"),
        "exp_type": exp_type,
        "groups": {
            group: {"fresh": value, "logged": logged.get(group)}
            for group, value in fresh.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="paper", help="results/<tag> to scan")
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="episodes per run (training-time eval used 20; more tightens the "
             "per-seed estimate and costs seconds)",
    )
    parser.add_argument("--only", default=None, help="comma-separated variants")
    parser.add_argument(
        "--out",
        default=None,
        help="CSV output path (default results/<tag>/checkpoint_eval.csv)",
    )
    args = parser.parse_args()

    root = Path(args.results_root) / args.tag
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1

    wanted = set(args.only.split(",")) if args.only else None
    variants = sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    if wanted:
        variants = [v for v in variants if v.name in wanted]

    out_path = Path(args.out) if args.out else root / "checkpoint_eval.csv"
    scratch = Path(tempfile.mkdtemp(prefix="ckpt_eval_"))
    rows: list[dict] = []

    try:
        for variant in variants:
            seeds = sorted(
                (d for d in variant.iterdir() if d.is_dir() and d.name.startswith("seed_")),
                key=lambda d: int(d.name.split("_")[1]),
            )
            for run_dir in seeds:
                print(f"[eval] {variant.name}/{run_dir.name}")
                result = _evaluate_run(run_dir, args.episodes, scratch)
                if result is None:
                    continue
                for group, vals in result["groups"].items():
                    rows.append(
                        {
                            "variant": result["variant"],
                            "seed": result["seed"],
                            "group": group,
                            "eval_reward_mean_fixed": vals["fresh"],
                            "eval_reward_mean_logged": vals["logged"],
                            "delta": (
                                None
                                if vals["logged"] is None
                                else vals["fresh"] - vals["logged"]
                            ),
                        }
                    )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not rows:
        print("nothing evaluated")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path.resolve()}")

    # ---- aggregate: mean +/- sem across seeds, the usual RL reporting unit ----
    print(f"\n{'variant':<16} {'n':>3}  {'re-eval (fixed)':>20}  "
          f"{'as logged':>20}  {'delta':>8}")
    print("-" * 74)
    for variant in variants:
        vals = [r for r in rows if r["variant"] == variant.name]
        if not vals:
            continue
        fresh = [r["eval_reward_mean_fixed"] for r in vals]
        logged = [r["eval_reward_mean_logged"] for r in vals
                  if r["eval_reward_mean_logged"] is not None]
        n = len(fresh)
        f_mean = mean(fresh)
        f_sem = (stdev(fresh) / n ** 0.5) if n > 1 else 0.0
        if logged:
            l_mean = mean(logged)
            l_sem = (stdev(logged) / len(logged) ** 0.5) if len(logged) > 1 else 0.0
            l_str = f"{l_mean:+9.3f} +/- {l_sem:5.3f}"
            d_str = f"{f_mean - l_mean:+8.3f}"
        else:
            l_str, d_str = " " * 20, " " * 8
        print(f"{variant.name:<16} {n:>3}  {f_mean:+9.3f} +/- {f_sem:5.3f}  "
              f"{l_str}  {d_str}")

    print(
        "\nre-eval (fixed) = old policy, fixed metric.  The gap to 'as logged' is "
        "the\nmeasurement bug alone; comparing against a fresh sweep isolates the "
        "training bug."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
