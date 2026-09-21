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

Protocols
---------
IBMARL has two evaluation protocols and this script can measure either:

  ``rl``     the RL actors alone -- what every run predating the protocol change
             logged, and the only thing the baselines have.
  ``rl_il``  the arbitrated RL+IL policy, i.e. the critic picking greedily
             between the teacher's action and the RL actor's.  Needs
             ``checkpoints/critic_checkpoint.pt`` and a readable teacher, so it
             is skipped with a warning for runs that predate the change or whose
             config points at a teacher path that no longer exists.

Usage
-----
    python -m analysis.eval_checkpoints                      # results/paper
    python -m analysis.eval_checkpoints --episodes 500
    python -m analysis.eval_checkpoints --only ibmarl,maddpg
    python -m analysis.eval_checkpoints --protocol rl_il
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

from src.experiments.base_marl_experiment import BaseMARLExperiment, vmas_rng_guard


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


def _build_ibmarl_eval_policies(cfg: dict, run_dir: Path, env, device, rl_policies):
    """
    Rebuild the arbitrated RL+IL policy for one IBMARL run, or None.

    Needs two artifacts beyond the actor checkpoint: the critic ensemble the
    arbiter scores with, and the frozen teacher it scores against.  Either can
    be absent -- runs predating the protocol change have no
    ``critic_checkpoint.pt``, and older configs carry absolute teacher paths from
    a machine or a directory layout that no longer exists (newer ones are
    repo-relative, ``teachers/<task>/...``, and resolve against this checkout)
    -- so both are checked and a miss is a warning, not a failure.
    """
    from src.experiments.ibmarl.arbiter import ActionArbiter
    from src.experiments.ibmarl.modules import build_eval_policies
    from src.experiments.ibmarl.networks import (
        R2bcPolicy,
        build_critics,
        build_targets,
    )

    critic_path = run_dir / "checkpoints" / "critic_checkpoint.pt"
    if not critic_path.exists():
        print(f"  skip rl_il for {run_dir}: no critic_checkpoint.pt")
        return None

    teacher_path = resolve_path(cfg.get("r2bc_checkpoint_path", ""))
    if not teacher_path.is_file():
        print(f"  skip rl_il for {run_dir}: teacher not readable at {teacher_path}")
        return None

    il_policy = R2bcPolicy(teacher_path, env, device)
    critics = build_critics(cfg, env, device)
    target_policies, target_critics = build_targets(rl_policies, critics, env)

    state = torch.load(critic_path, map_location=device, weights_only=False)
    missing = set(target_critics) - set(state.get("target_critics", {}))
    if missing:
        raise RuntimeError(f"{run_dir}: critic checkpoint missing groups {missing}")
    for group, ensemble in target_critics.items():
        ensemble.load_state_dict(state["target_critics"][group])
    for group, ensemble in critics.items():
        if group in state.get("critics", {}):
            ensemble.load_state_dict(state["critics"][group])

    # target_rl_policy is only read by the bootstrap path, which greedy
    # arbitration never enters; pass the real thing anyway so nothing quietly
    # depends on the argument being unused.
    arbiter = ActionArbiter(
        cfg,
        il_policy,
        rl_policies,
        target_policies,
        critics,
        target_critics,
        env,
        device,
        il_noise_modules=None,
    )
    return build_eval_policies(arbiter, rl_policies, env)


def _last_logged_eval(run_dir: Path) -> tuple[dict, str | None]:
    """
    Final logged ``eval_reward_mean`` per group, i.e. column (a), plus the
    protocol it was measured under.

    ``eval_protocol`` was added with the RL+IL protocol; a run without that
    column predates the change and its ``eval_reward_mean`` is RL-only.  Without
    this the ``delta`` below would silently compare an RL-only re-eval against an
    RL+IL logged value.
    """
    metrics = run_dir / "data" / "metrics.csv"
    if not metrics.exists():
        return {}, None
    latest: dict[str, float] = {}
    protocol = None
    with metrics.open() as f:
        for row in csv.DictReader(f):
            raw = (row.get("eval_reward_mean") or "").strip()
            if raw:
                try:
                    latest[row["group"]] = float(raw)
                except ValueError:
                    continue
                protocol = (row.get("eval_protocol") or "").strip() or "rl"
    return latest, protocol


def _evaluate_run(
    run_dir: Path, n_episodes: int, scratch: Path, protocol: str
) -> dict | None:
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

    eval_policies = None
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

        if exp_type == "ibmarl" and protocol in ("rl_il", "both"):
            eval_policies = _build_ibmarl_eval_policies(
                cfg, run_dir, exp.env, exp.device, policies
            )

    # Same seed for both protocols so their difference is paired, and guarded so
    # a re-eval cannot disturb anything else sharing VMAS's RNG stream.
    seed = int(cfg.get("seed", 0)) + 10_000
    fresh_rl: dict = {}
    fresh_rl_il: dict = {}
    with vmas_rng_guard():
        if protocol in ("rl", "both") or eval_policies is None:
            exp.reseed_eval_env(seed)
            fresh_rl = exp.evaluate(n_episodes=n_episodes)
        if eval_policies is not None:
            exp.reseed_eval_env(seed)
            fresh_rl_il = exp.evaluate(n_episodes=n_episodes, policies=eval_policies)

    logged, logged_protocol = _last_logged_eval(run_dir)
    groups = set(fresh_rl) | set(fresh_rl_il)

    return {
        "variant": run_dir.parent.name,
        "seed": cfg.get("seed"),
        "exp_type": exp_type,
        "groups": {
            group: {
                "rl": fresh_rl.get(group),
                "rl_il": fresh_rl_il.get(group),
                "logged": logged.get(group),
                "logged_protocol": logged_protocol,
            }
            for group in groups
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
        "--protocol",
        choices=("rl", "rl_il", "both"),
        default="both",
        help="which evaluation protocol(s) to measure; rl_il applies to ibmarl "
             "runs only and needs a critic checkpoint plus a readable teacher",
    )
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
                result = _evaluate_run(
                    run_dir, args.episodes, scratch, args.protocol
                )
                if result is None:
                    continue
                for group, vals in result["groups"].items():
                    # delta compares like with like: the logged column is RL-only
                    # for runs predating the protocol change and RL+IL after it.
                    same = vals.get(vals["logged_protocol"] or "rl")
                    rows.append(
                        {
                            "variant": result["variant"],
                            "seed": result["seed"],
                            "group": group,
                            "eval_reward_mean_rl": vals["rl"],
                            "eval_reward_mean_rl_il": vals["rl_il"],
                            # Back-compat alias for readers written against the
                            # single-protocol CSV (analysis/paper3_figures.py):
                            # whichever protocol matches what the run logged.
                            "eval_reward_mean_fixed": same,
                            "eval_reward_mean_logged": vals["logged"],
                            "logged_protocol": vals["logged_protocol"],
                            "delta": (
                                None
                                if vals["logged"] is None or same is None
                                else same - vals["logged"]
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
    def _stat(values):
        vals = [v for v in values if v is not None]
        if not vals:
            return None, None
        n = len(vals)
        return mean(vals), (stdev(vals) / n ** 0.5) if n > 1 else 0.0

    def _fmt(m, sem):
        return " " * 20 if m is None else f"{m:+9.3f} +/- {sem:5.3f}"

    print(f"\n{'variant':<16} {'n':>3}  {'re-eval RL-only':>20}  "
          f"{'re-eval RL+IL':>20}  {'as logged':>20}  {'delta':>8}")
    print("-" * 96)
    for variant in variants:
        vals = [r for r in rows if r["variant"] == variant.name]
        if not vals:
            continue
        rl_mean, rl_sem = _stat(r["eval_reward_mean_rl"] for r in vals)
        il_mean, il_sem = _stat(r["eval_reward_mean_rl_il"] for r in vals)
        l_mean, l_sem = _stat(r["eval_reward_mean_logged"] for r in vals)
        d_mean, _ = _stat(r["delta"] for r in vals)
        d_str = " " * 8 if d_mean is None else f"{d_mean:+8.3f}"
        print(f"{variant.name:<16} {len(vals):>3}  {_fmt(rl_mean, rl_sem)}  "
              f"{_fmt(il_mean, il_sem)}  {_fmt(l_mean, l_sem)}  {d_str}")

    print(
        "\nre-eval = old policy, fixed metric.  'delta' compares the re-eval "
        "against\n'as logged' on the protocol that run actually logged (see "
        "logged_protocol);\nthe gap is the measurement bug alone, while comparing "
        "against a fresh sweep\nisolates the training bug.  RL+IL is blank for "
        "baselines and for runs with no\ncritic checkpoint."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
