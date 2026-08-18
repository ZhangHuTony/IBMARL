"""
Single (variant, seed) run for the workshop-paper sweep.

Writes into ``results/paper/<variant>/seed_<n>/`` so that
``analysis/main_results.py``-style seed discovery works out of the box
(each seed dir contains ``data/metrics.csv``).

Run from the repository root:
    python -m analysis.paper_run --variant ibmarl --seed 0
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

from src.run_experiment import load_config
from src.experiment_registry import EXPERIMENT_REGISTRY


# variant name -> (exp_type, config overrides applied on top of the merged yaml)
VARIANTS: dict[str, tuple[str, dict]] = {
    # --- main comparison ---
    "maddpg": ("maddpg", {}),
    "rlfd": ("rlfd", {}),
    "rft": ("rft", {}),
    "ibmarl": ("ibmarl", {}),
    # --- IBMARL ablations (deltas from the default ibmarl config) ---
    # all-IL vs all-RL joint action instead of the per-agent 2^N combination
    "ibmarl_strict": ("ibmarl", {"strict": True}),
    # greedy argmax over candidates instead of Boltzmann sampling
    "ibmarl_hard": ("ibmarl", {"soft": False}),
    # single critic instead of the 3-member ensemble
    "ibmarl_1critic": ("ibmarl", {"num_critics": 1}),
    # --- reference ---
    # bc_eval.yaml still points at a collaborator's home dir; reuse the same
    # R2BC checkpoint the demo-based methods are given.
    "bc_eval": ("bc_eval", {"__use_ibmarl_bc_checkpoint__": True}),
}


def _ibmarl_bc_checkpoint() -> str:
    with open(Path("config") / "experiments" / "ibmarl.yaml") as f:
        return yaml.safe_load(f)["r2bc_checkpoint_path"]


def build_cfg(
    variant: str,
    scenario: str,
    seed: int,
    n_iters: int | None,
    n_opt_steps: int | None = None,
    tag: str | None = None,
    resume_interval: int | None = None,
) -> dict:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}. Known: {sorted(VARIANTS)}")
    exp_type, overrides = VARIANTS[variant]

    cfg = load_config(scenario, exp_type)
    overrides = dict(overrides)
    if overrides.pop("__use_ibmarl_bc_checkpoint__", False):
        cfg["r2bc_checkpoint_path"] = _ibmarl_bc_checkpoint()
    cfg.update(overrides)
    cfg["seed"] = seed
    cfg["render"] = False
    if n_iters is not None:
        cfg["n_iters"] = n_iters
        cfg["total_frames"] = cfg["frames_per_batch"] * n_iters
    if n_opt_steps is not None:
        cfg["training"]["n_optimiser_steps"] = n_opt_steps
    if resume_interval is not None:
        cfg["resume_interval"] = resume_interval

    root = Path("results") / (tag or "paper")
    run_dir = root / variant / f"seed_{seed}"
    for sub in ("data", "checkpoints", "videos", "plots"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    cfg["variant"] = variant
    cfg["run_name"] = f"{variant}_seed_{seed}"
    cfg["run_dir"] = str(run_dir)
    cfg["data_dir"] = str(run_dir / "data")
    cfg["check_dir"] = str(run_dir / "checkpoints")
    cfg["videos_dir"] = str(run_dir / "videos")
    cfg["plots_dir"] = str(run_dir / "plots")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--scenario", default="navigation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="Override n_iters (used for smoke tests).",
    )
    parser.add_argument(
        "--n-opt-steps",
        type=int,
        default=None,
        help="Override training.n_optimiser_steps (used for smoke tests).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Results subdirectory under results/ (default: 'paper').",
    )
    parser.add_argument(
        "--resume-interval",
        type=int,
        default=None,
        help="Iterations between mid-run resume checkpoints (default 25).",
    )
    args = parser.parse_args()

    cfg = build_cfg(
        args.variant,
        args.scenario,
        args.seed,
        args.n_iters,
        args.n_opt_steps,
        args.tag,
        args.resume_interval,
    )
    run_dir = Path(cfg["run_dir"])

    status = {"variant": args.variant, "seed": args.seed, "pid": os.getpid()}
    t0 = time.time()
    try:
        experiment = EXPERIMENT_REGISTRY[cfg["exp_type"]](cfg)
        summary = experiment.train()
        experiment.save_results()
        status.update(ok=True, summary=summary)
        print(summary)
    except Exception:
        status.update(ok=False, error=traceback.format_exc())
        traceback.print_exc()
    finally:
        status["wall_seconds"] = round(time.time() - t0, 1)
        with open(run_dir / "status.json", "w") as f:
            json.dump(status, f, indent=2)

    return 0 if status.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
