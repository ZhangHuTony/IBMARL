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


def _gated(alpha: float, mode: str = "gated", *,
           strict: bool = True, **extra) -> tuple[str, dict]:
    """
    IBMARL plus the gated imitation term at *alpha* (config actor_reg; see
    CONTEXT.md).  The override is the whole ``actor_reg`` dict because
    build_cfg applies overrides with a flat ``cfg.update``.  Variants are named
    by alpha so a pilot seed is reused, not re-run, once alpha is chosen.
    ``mode="uniform"`` is the gate-held-open control.  ``strict`` selects the
    base method: True for the joint all-IL/all-RL choice, False for per-agent
    mixing.  ``**extra`` carries further one-factor deltas (soft, num_critics)
    into the same flat override dict.

    ``temperature`` is deliberately ABSENT from the actor_reg dict.  losses.py
    reads it with ``reg.get("temperature")`` -> None, and arbiter.py then falls
    back to the arbiter's own ``temperature``, so the gate is always scored at
    the temperature execution-time arbitration uses.  Pinning a number here
    instead silently desyncs the two on any task that overrides the arbiter
    temperature: transport pins 0.05 (ibmarl_transport.yaml) against
    ibmarl.yaml's 0.0005, so a hard-coded 0.0005 gave it a gate 100x too sharp.
    The flat ``cfg.update`` is what makes the omission effective -- it discards
    ibmarl.yaml's actor_reg block wholesale, its 0.0005 included.  The
    buzzwire3_reg runs on disk carry an explicit 0.05 in both slots; they were
    frozen before the arbiter temperature was rescaled, and satisfy the same
    invariant (gate temperature == arbiter temperature), not the same number.
    """
    cfg = {"strict": strict,
           "actor_reg": {"mode": mode, "alpha": alpha, "gate": "soft"}}
    cfg.update(extra)
    return ("ibmarl", cfg)


# --- actor-lag experiment (results/buzzwire3_reg; not in paper_sweep.JOBS) ---
VARIANTS.update({f"ibmarl_strict_gated_a{a:g}": _gated(a) for a in (0.1, 0.4, 1.6)})

# The gate-held-open control at the chosen alpha: same pull toward the teacher,
# applied at every observation instead of only where the arbiter prefers the
# teacher.  It separates "gating by the critic" from "any imitation term at
# all", and is the in-method twin of the RFT baseline (whose BC term is
# ungated and annealed on a clock rather than by the critic).
VARIANTS.update({f"ibmarl_strict_uniform_a{a:g}": _gated(a, mode="uniform")
                 for a in (0.4,)})

# --- the main method, as of 2026-09-20 (results/transport800) ------------
# IBMARL is per-agent MIXING plus the gated imitation term.  `strict=False`
# restates ibmarl.yaml's default explicitly so the frozen config.yaml is
# unambiguous and a later change to that default cannot silently redefine the
# method.  Its one-factor ablations sit beside it; `ibmarl_strict_gated_a0.4`
# above doubles as the "w/o mixing" arm and plain `ibmarl` as the "w/o
# regularisation" arm, so neither needs a new name.
VARIANTS["ibmarl_gated_a0.4"] = _gated(0.4, strict=False)
# w/o critic ensemble: one critic instead of three, in arbitration and in the
# gate alike (both go through ActionArbiter._sample_critic_indices).  Also
# answers the standing objection that IBMARL gets 3 critics to each baseline's 1.
VARIANTS["ibmarl_gated_a0.4_1critic"] = _gated(0.4, strict=False, num_critics=1)

# --- human-teacher arms (results/buzzwire6; poster, 2026-09-24) -------------
# The same methods with the human Xbox teacher in place of the heuristic
# demonstrator's (teachers/human_buzz_wire_24_demos, see teachers/README.md).
# That recording is the legacy -1/step schema, so these run on the frozen
# legacy bases (--base-config legacy); demonstrations_legacy.pt is the
# recording with its 24 time-limit ends unflagged as terminals
# (analysis/prepare_human_demos.py).  The overrides ride the flat cfg.update
# like every other variant, so they apply on any base.
_HUMAN_BW = {
    "r2bc_checkpoint_path": "teachers/human_buzz_wire_24_demos/policy_checkpoint.pth",
    "demonstrations_path": "teachers/human_buzz_wire_24_demos/demonstrations_legacy.pt",
}
VARIANTS["bc_eval_human"] = ("bc_eval", dict(_HUMAN_BW))
VARIANTS["rlfd_human"] = ("rlfd", dict(_HUMAN_BW))
VARIANTS["rft_human"] = ("rft", dict(_HUMAN_BW))
VARIANTS["ibmarl_gated_a0.4_human"] = _gated(0.4, strict=False, **_HUMAN_BW)

# Keys build_cfg (re)assigns for every run; a frozen config.yaml carries its
# original run's values for them and they must not leak into a new run.
RUN_KEYS = ("variant", "run_name", "run_dir", "data_dir", "check_dir",
            "videos_dir", "plots_dir", "seed")


def load_base_config(base_config: str, scenario: str, exp_type: str) -> dict:
    """
    A frozen ``config.yaml`` as the base of a run, in place of the live yaml
    stack (``load_config``).  ``"legacy"`` selects
    ``config/legacy/<scenario>/<exp_type>.yaml``, the bases copied from the
    sweeps the poster reuses (paper4, buzzwire3); anything else is a path.

    Why: the live stack moves (reward schema, arbiter temperature, teacher,
    replay size, eval cadence), so a new arm launched from it is not
    comparable with the runs already on disk.  A frozen base is a bit-for-bit
    copy of what those runs saw, and the variant's overrides are applied on
    top exactly as they are on the live stack.  The scenario and exp_type
    embedded in the file must match the run, or a buzz-wire base could
    silently drive a navigation run.
    """
    if base_config == "legacy":
        path = Path("config/legacy") / scenario / f"{exp_type}.yaml"
    else:
        path = Path(base_config)
    if not path.exists():
        raise FileNotFoundError(f"--base-config: {path} does not exist")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key in RUN_KEYS:
        cfg.pop(key, None)
    for key, want in (("scenario_name", scenario), ("exp_type", exp_type)):
        have = cfg.get(key)
        if have != want:
            raise ValueError(
                f"--base-config {path}: {key}={have!r}, but this run is {want!r}"
            )
    cfg["base_config"] = str(path)
    print(f"[paper_run] base config: {path}")
    return cfg


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
    base_config: str | None = None,
) -> dict:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown variant {variant!r}. Known: {sorted(VARIANTS)}")
    exp_type, overrides = VARIANTS[variant]

    if base_config:
        cfg = load_base_config(base_config, scenario, exp_type)
    else:
        cfg = load_config(scenario, exp_type)
    cfg.update(overrides)
    cfg["seed"] = seed
    cfg["render"] = False
    if n_iters is not None:
        # RFT's BC term anneals over `rft.bc_anneal_n_iters` iterations, which is
        # only meaningful as a FRACTION of the budget -- the overlays say so in
        # capitals (config/experiments/rft_transport.yaml) and the project has
        # got it wrong in both directions.  --n-iters silently left it behind,
        # so an 800-iteration overlay under a 1200-iteration run would anneal
        # over 67% instead of the 100% it was set for, quietly making RFT a
        # different method.  Rescale it by the same factor as the budget, which
        # preserves whatever ratio the overlay actually encodes rather than
        # assuming half or full.
        base_iters = cfg.get("n_iters")
        rft_cfg = cfg.get("rft")
        if (isinstance(rft_cfg, dict) and rft_cfg.get("bc_anneal_n_iters")
                and base_iters):
            scaled = round(int(rft_cfg["bc_anneal_n_iters"]) * n_iters / int(base_iters))
            if scaled != rft_cfg["bc_anneal_n_iters"]:
                print(f"[paper_run] budget {base_iters} -> {n_iters}: "
                      f"rft.bc_anneal_n_iters {rft_cfg['bc_anneal_n_iters']} -> {scaled} "
                      f"(ratio {int(rft_cfg['bc_anneal_n_iters'])/int(base_iters):.2f} kept)")
            cfg["rft"] = dict(rft_cfg, bc_anneal_n_iters=scaled)
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
    parser.add_argument(
        "--base-config",
        default=None,
        help="Frozen config.yaml to build the run from instead of the live yaml "
             "stack; 'legacy' = config/legacy/<scenario>/<exp_type>.yaml.",
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
        args.base_config,
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
