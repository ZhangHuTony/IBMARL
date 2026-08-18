"""
Orchestrator for the workshop-paper sweep.

Runs every (variant, seed) job at a bounded concurrency, and is itself
resumable at two levels:

  * run level    -- a job whose ``status.json`` says ok is skipped entirely.
  * iteration    -- an interrupted job resumes mid-run from its last
                    ``checkpoints/resume`` point (see src/util/checkpointing.py).

So re-running this script after any interruption picks up where it left off.

    python -m analysis.paper_sweep --jobs 3            # launch / resume
    python -m analysis.paper_sweep --status            # progress only
    python -m analysis.paper_sweep --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RESULTS_ROOT = Path("results")

# (variant, [seeds])
MAIN_SEEDS = [0, 1, 2, 3, 4]
ABLATION_SEEDS = [0, 1, 2]

JOBS: list[tuple[str, list[int]]] = [
    # IL reference line: no training, seconds per seed. Runs first so the
    # reference number exists on disk before any learning curve does.
    ("bc_eval", ABLATION_SEEDS),
    ("ibmarl", MAIN_SEEDS),
    ("maddpg", MAIN_SEEDS),
    ("rlfd", MAIN_SEEDS),
    ("rft", MAIN_SEEDS),
    ("ibmarl_strict", ABLATION_SEEDS),
    ("ibmarl_hard", ABLATION_SEEDS),
    ("ibmarl_1critic", ABLATION_SEEDS),
]


def job_list(
    only: list[str] | None, max_seeds: int | None = None
) -> list[tuple[str, int]]:
    jobs = []
    for variant, seeds in JOBS:
        if only and variant not in only:
            continue
        for seed in seeds[:max_seeds] if max_seeds else seeds:
            jobs.append((variant, seed))
    return jobs


def run_dir(tag: str, variant: str, seed: int) -> Path:
    return RESULTS_ROOT / tag / variant / f"seed_{seed}"


def job_state(tag: str, variant: str, seed: int) -> str:
    """One of: done, partial, pending."""
    d = run_dir(tag, variant, seed)
    status = d / "status.json"
    if status.exists():
        try:
            if json.load(open(status)).get("ok"):
                return "done"
        except (json.JSONDecodeError, OSError):
            pass
    if (d / "checkpoints" / "resume" / "state.pt").exists():
        return "partial"
    return "pending"


def resume_iteration(tag: str, variant: str, seed: int) -> int | None:
    p = run_dir(tag, variant, seed) / "checkpoints" / "resume" / "state.pt"
    if not p.exists():
        return None
    try:
        import torch

        return int(torch.load(p, map_location="cpu", weights_only=False)["iteration"])
    except Exception:
        return None


def print_status(tag: str, only: list[str] | None, max_seeds: int | None = None) -> None:
    jobs = job_list(only, max_seeds)
    counts = {"done": 0, "partial": 0, "pending": 0}
    print(f"{'variant':<16} {'seed':>4}  {'state':<8} {'resume@':>8}")
    print("-" * 42)
    for variant, seed in jobs:
        st = job_state(tag, variant, seed)
        counts[st] += 1
        it = resume_iteration(tag, variant, seed) if st == "partial" else None
        print(f"{variant:<16} {seed:>4}  {st:<8} {'' if it is None else it:>8}")
    print("-" * 42)
    print(
        f"{counts['done']} done, {counts['partial']} partial, "
        f"{counts['pending']} pending  (of {len(jobs)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3, help="concurrent runs")
    parser.add_argument("--threads", type=int, default=4, help="torch threads per run")
    parser.add_argument("--tag", default="paper")
    parser.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="Override n_iters. Default: leave it to config/base.yaml.",
    )
    parser.add_argument("--resume-interval", type=int, default=25)
    parser.add_argument("--only", default=None, help="comma-separated variants")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Cap seeds per variant (smoke tests).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument(
        "--force", action="store_true", help="re-run jobs already marked done"
    )
    args = parser.parse_args()

    only = args.only.split(",") if args.only else None

    if args.status:
        print_status(args.tag, only, args.max_seeds)
        return 0

    jobs = job_list(only, args.max_seeds)
    todo = [
        (v, s)
        for v, s in jobs
        if args.force or job_state(args.tag, v, s) != "done"
    ]

    print(f"{len(jobs)} jobs total, {len(todo)} to run at concurrency {args.jobs}")
    for v, s in todo:
        st = job_state(args.tag, v, s)
        it = resume_iteration(args.tag, v, s)
        extra = f" (resumes at iter {it + 1})" if it is not None else ""
        print(f"  {v:<16} seed {s}  [{st}]{extra}")
    if args.dry_run:
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    log_dir = RESULTS_ROOT / args.tag / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
    )

    running: list[tuple[subprocess.Popen, str, int, float]] = []
    queue = list(todo)
    completed: list[tuple[str, int, bool, float]] = []
    interrupted = False

    def shutdown(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[sweep] interrupt received -- stopping child runs; "
              "their resume points are on disk. Re-run this script to continue.")
        for proc, _, _, _ in running:
            proc.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    t_start = time.time()
    while (queue or running) and not interrupted:
        while queue and len(running) < args.jobs and not interrupted:
            variant, seed = queue.pop(0)
            log_path = log_dir / f"{variant}_seed_{seed}.log"
            cmd = [
                sys.executable,
                "-m",
                "analysis.paper_run",
                "--variant", variant,
                "--seed", str(seed),
                "--tag", args.tag,
                "--resume-interval", str(args.resume_interval),
            ]
            if args.n_iters is not None:
                cmd += ["--n-iters", str(args.n_iters)]
            fh = open(log_path, "a")
            fh.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running.append((proc, variant, seed, time.time()))
            print(f"[sweep] started {variant} seed {seed} (pid {proc.pid}) -> {log_path}")

        time.sleep(5)

        still = []
        for proc, variant, seed, t0 in running:
            if proc.poll() is None:
                still.append((proc, variant, seed, t0))
                continue
            ok = proc.returncode == 0
            dt = time.time() - t0
            completed.append((variant, seed, ok, dt))
            n_done = len(completed)
            print(
                f"[sweep] finished {variant} seed {seed} "
                f"{'OK' if ok else 'FAILED rc=%d' % proc.returncode} "
                f"in {dt/60:.1f} min  ({n_done}/{len(todo)} done, "
                f"{(time.time()-t_start)/60:.0f} min elapsed)"
            )
        running = still

    for proc, _, _, _ in running:
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()

    n_ok = sum(1 for _, _, ok, _ in completed if ok)
    print(
        f"\n[sweep] {n_ok}/{len(completed)} runs succeeded in "
        f"{(time.time()-t_start)/60:.0f} min"
    )
    if interrupted:
        print("[sweep] interrupted -- re-run to resume.")
        return 130
    print_status(args.tag, only, args.max_seeds)
    return 0 if n_ok == len(completed) else 1


if __name__ == "__main__":
    sys.exit(main())
