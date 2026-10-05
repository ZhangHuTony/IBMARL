"""
Pre-flight for a sweep task: build the run config, and generate any DERIVED
demonstration file it needs that is not on disk yet.

Two artifacts the configs point at are derived from bundled recordings and need
torch to make, which the workstation that edits the configs does not have:

  teachers/balance/demonstrations_sparse.pt          analysis/relabel_demos_sparse.py
      (the dense balance recording relabelled at balance.yaml's gt_radius)
  teachers/human_transport_90_demos/demonstrations_legacy.pt
      analysis/prepare_human_demos.py (time-limit ends unflagged as terminals)

Rather than gate the batch on a commit from a torch machine, every array task of
slurm/run_balance.slurm runs this first.  Generation is idempotent and
deterministic, written to a temp file beside the target and renamed into place
(atomic on POSIX), and the SLURM script wraps the call in ``flock`` so tasks
that start together do not write at once.  A generated balance file whose
``meta.radius`` disagrees with the config's gt_radius is regenerated, so a
radius change in the yaml cannot silently run against stale demo labels.

Then the config's artifact paths are resolved and asserted to exist, so a
missing overlay (e.g. rft_balance.yaml) or checkpoint fails here, before the GPU
has been held for an hour.  Writes only results/_precheck/ (build_cfg's run dir).

    python -m analysis.ensure_demo_artifacts <variant> <scenario>

The generated files are reproducible from the recording + config, so they need
not be committed; commit them anyway (git add teachers/) if you want the
checkout alone to be enough on a machine without torch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from analysis.paper_run import build_cfg
from src.util.paths import PROJECT_ROOT, resolve_path

SPARSE_BALANCE = PROJECT_ROOT / "teachers" / "balance" / "demonstrations_sparse.pt"
HUMAN_TRANSPORT_LEGACY = PROJECT_ROOT / "teachers" / "human_transport_90_demos" / "demonstrations_legacy.pt"


def _tmp_beside(target: Path) -> Path:
    return target.with_name(f"{target.name}.tmp-{os.uname().nodename if hasattr(os, 'uname') else 'host'}-{os.getpid()}")


def _file_radius(path: Path) -> float | None:
    try:
        meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
        return float(meta.get("radius"))
    except Exception as e:  # unreadable / half-written -> regenerate
        print(f"[ensure] {path.name}: cannot read meta ({type(e).__name__}); regenerating")
        return None


def ensure_sparse_balance(radius: float) -> None:
    if SPARSE_BALANCE.exists():
        have = _file_radius(SPARSE_BALANCE)
        if have is not None and abs(have - radius) < 1e-9:
            print(f"[ensure] {SPARSE_BALANCE.relative_to(PROJECT_ROOT)} present (radius {have:g})")
            return
        print(f"[ensure] {SPARSE_BALANCE.name} was built at radius {have}, config says {radius:g}: regenerating")
    from analysis.relabel_demos_sparse import relabel
    tmp = _tmp_beside(SPARSE_BALANCE)
    try:
        relabel("balance", SPARSE_BALANCE.with_name("demonstrations.pt"), tmp, radius, [0.3, 0.5, 0.75, 1.05])
        os.replace(tmp, SPARSE_BALANCE)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"[ensure] wrote {SPARSE_BALANCE.relative_to(PROJECT_ROOT)} at radius {radius:g}")


def ensure_human_transport() -> None:
    if HUMAN_TRANSPORT_LEGACY.exists():
        print(f"[ensure] {HUMAN_TRANSPORT_LEGACY.relative_to(PROJECT_ROOT)} present")
        return
    from analysis.prepare_human_demos import prepare
    tmp = _tmp_beside(HUMAN_TRANSPORT_LEGACY)
    try:
        prepare("transport", dst=tmp)
        os.replace(tmp, HUMAN_TRANSPORT_LEGACY)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"[ensure] wrote {HUMAN_TRANSPORT_LEGACY.relative_to(PROJECT_ROOT)}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    variant, scenario = argv[1:3]
    cfg = build_cfg(variant, scenario, 0, None, tag="_precheck")

    demos = cfg.get("demonstrations_path")
    if demos and resolve_path(demos) == SPARSE_BALANCE:
        ensure_sparse_balance(float(cfg["gt_radius"]))
    if demos and resolve_path(demos) == HUMAN_TRANSPORT_LEGACY:
        ensure_human_transport()

    for key in ("r2bc_checkpoint_path", "demonstrations_path"):
        if key in cfg:
            p = resolve_path(cfg[key])
            if not p.exists():
                print(f"[pre-check] {key} -> {p} does not exist")
                return 1
            print(f"[pre-check] {key}: {cfg[key]}")
    print(f"[pre-check] {scenario}/{variant}: n_iters {cfg.get('n_iters')}, total_frames {cfg.get('total_frames')}, "
          f"sparse_rewards {cfg.get('sparse_rewards')}, suppress_done {cfg.get('suppress_done')}, "
          f"gt_radius {cfg.get('gt_radius')}, human_teacher {cfg.get('human_teacher')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
