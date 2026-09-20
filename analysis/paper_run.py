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
    # one-factor ablations of the strict (joint-choice) main method
    "ibmarl_strict_hard": ("ibmarl", {"strict": True, "soft": False}),
    "ibmarl_strict_1critic": ("ibmarl", {"strict": True, "num_critics": 1}),
    # --- reference ---
    # bc_eval*.yaml carry the same bundled teacher (teachers/<task>/) the
    # demo-based methods bootstrap from, so the reference line needs no override.
    "bc_eval": ("bc_eval", {}),
}


def _gated(alpha: float, mode: str = "gated") -> tuple[str, dict]:
    """
    IBMARL (strict) plus the gated imitation term at *alpha* (config
    actor_reg; see CONTEXT.md).  The override is the whole ``actor_reg`` dict
    because build_cfg applies overrides with a flat ``cfg.update``.  Variants
    are named by alpha so a pilot seed is reused, not re-run, once alpha is
    chosen.  ``mode="uniform"`` is the gate-held-open control, registered
    separately below under an ``ibmarl_strict_uniform_a*`` name.
    """
    return ("ibmarl", {"strict": True, "actor_reg": {
        # Gate temperature = the arbiter's, rescaled for the binary terminal
        # schema (see ibmarl.yaml).  These variants are the buzz_wire actor-lag
        # experiment; a legacy-schema task would need 0.05 here.
        "mode": mode, "alpha": alpha, "gate": "soft", "temperature": 0.0005}})


# --- actor-lag experiment (results/buzzwire3_reg; not in paper_sweep.JOBS) ---
VARIANTS.update({f"ibmarl_strict_gated_a{a:g}": _gated(a) for a in (0.1, 0.4, 1.6)})

# The gate-held-open control at the chosen alpha: same pull toward the teacher,
# applied at every observation instead of only where the arbiter prefers the
# teacher.  It separates "gating by the critic" from "any imitation term at
# all", and is the in-method twin of the RFT baseline (whose BC term is
# ungated and annealed on a clock rather than by the critic).
VARIANTS.update({f"ibmarl_strict_uniform_a{a:g}": _gated(a, mode="uniform")
                 for a in (0.4,)})


def build_cfg(
    variant: str,
    scenario: str,
    seed: int,
    n_iters: int | None,
    n_opt_steps: int | None = None,
    tag: str | None = None,
    resume_interval: int | None = None,
    sigma_init: float | None = None,
    sigma_end: float | None = None,
) -> dict:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}. Known: {sorted(VARIANTS)}")
    exp_type, overrides = VARIANTS[variant]

    cfg = load_config(scenario, exp_type)
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
    # Exploration-noise overrides, mirroring src.run_experiment's --sigma_init/--sigma_end.
    # Both the RL actor's noise and the teacher-proposal noise read this same block
    # (ibmarl/modules.py), so these anneal each of them.
    if sigma_init is not None or sigma_end is not None:
        noise = dict(cfg.get("exploration_noise") or {})
        if sigma_init is not None:
            noise["sigma_init"] = sigma_init
        if sigma_end is not None:
            noise["sigma_end"] = sigma_end
        cfg["exploration_noise"] = noise

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
        "--sigma-init",
        type=float,
        default=None,
        help="Override exploration_noise.sigma_init (IBMARL variants).",
    )
    parser.add_argument(
        "--sigma-end",
        type=float,
        default=None,
        help="Override exploration_noise.sigma_end (IBMARL variants).",
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
        args.sigma_init,
        args.sigma_end,
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
