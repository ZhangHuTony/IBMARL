"""
Prepare the human Xbox recordings for the demonstration loader.

``teachers/human_*_demos/demonstrations.pt`` (recorded 2026-09-23 with
src/r2bc/human_teleop.py on the collaborator's machine, format_version 2) flag
EVERY episode end as ``dones=True, terminated=True`` -- including the ends that
are plain time limits.  ``src/util/demonstrations.py`` reads ``terminated`` as
"true terminal", so loaded as-is the critic would stop bootstrapping at every
time-limit end: 24 of 24 buzz-wire episodes, 85 of 90 transport episodes.

This writes ``demonstrations_legacy.pt`` beside each recording with the flags
corrected and everything else byte-identical; the raw recording is never
touched.

  buzz_wire   legacy -1/step schema, fixed 200-step horizon with done()
              suppressed: no episode end is a terminal.  ``dones`` and
              ``terminated`` are cleared, matching the heuristic recordings
              (all-False ``dones``, no ``terminated`` key).
  transport   dense reward; VMAS ends the episode when the package is
              delivered, so an episode SHORTER than the horizon ended on a
              true terminal.  ``dones``/``terminated`` are kept at those ends
              (5 of 90) and cleared at the 85 time-limit ends.

    python -m analysis.prepare_human_demos            # both tasks
    python -m analysis.prepare_human_demos buzz_wire
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.util.paths import PROJECT_ROOT

TASKS = {
    "buzz_wire": dict(dir="teachers/human_buzz_wire_24_demos", horizon=200, terminal="never"),
    "transport": dict(dir="teachers/human_transport_90_demos", horizon=500, terminal="early_end"),
}


def episode_bounds(dones: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_inclusive)] from the recorder's per-episode done flags."""
    ends = np.flatnonzero(dones)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return list(zip(starts.tolist(), ends.tolist()))


def prepare(task: str) -> dict:
    spec = TASKS[task]
    src = PROJECT_ROOT / spec["dir"] / "demonstrations.pt"
    dst = PROJECT_ROOT / spec["dir"] / "demonstrations_legacy.pt"
    d = torch.load(src, map_location="cpu", weights_only=False)
    meta = d.get("meta", {})
    assert meta.get("format_version") == 2, f"{src}: unexpected format {meta.get('format_version')}"
    dones = np.asarray(d["dones"], dtype=bool)
    terminated = np.asarray(d["terminated"], dtype=bool)
    assert dones.tolist() == terminated.tolist(), "recorder flags every end as terminal; layout changed?"
    bounds = episode_bounds(dones)
    assert bounds and bounds[-1][1] == len(dones) - 1, "last row is not an episode end"
    lengths = [e - s + 1 for s, e in bounds]
    assert max(lengths) <= spec["horizon"], (max(lengths), spec["horizon"])
    # Cross-check against the recorder's own episode list.
    rec = [ep["length"] for ep in meta.get("episodes", [])]
    assert rec == lengths, f"episode lengths disagree with meta: {rec[:5]}... vs {lengths[:5]}..."

    new_dones = np.zeros_like(dones)
    new_term = np.zeros_like(terminated)
    kept = 0
    if spec["terminal"] == "early_end":
        for (s, e), n in zip(bounds, lengths):
            if n < spec["horizon"]:
                new_dones[e] = True
                new_term[e] = True
                kept += 1
    obs = np.asarray(d["obs"])
    # Rows within an episode must chain (next_obs[t] == obs[t+1]); guards
    # against a mis-read layout before anything is written.
    for s, e in bounds[:3]:
        nxt = np.asarray(d["next_obs"])[s:e]
        assert np.allclose(nxt, obs[s + 1:e + 1]), "episodes are not contiguous"

    out = dict(d)
    out["dones"] = torch.as_tensor(new_dones)
    out["terminated"] = torch.as_tensor(new_term)
    out["meta"] = dict(meta, prepared_from=str(src.relative_to(PROJECT_ROOT)),
                       note=(f"analysis/prepare_human_demos.py: {len(bounds)} episodes; "
                             f"{len(bounds) - kept} time-limit ends unflagged, {kept} true "
                             f"terminals kept ({spec['terminal']})"))
    torch.save(out, dst)
    summary = dict(task=task, episodes=len(bounds), transitions=len(dones), lengths_min=min(lengths),
                   lengths_max=max(lengths), terminals_kept=kept, dst=str(dst.relative_to(PROJECT_ROOT)))
    print(summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", choices=sorted(TASKS), default=None)
    args = ap.parse_args()
    for task in ([args.task] if args.task else sorted(TASKS)):
        prepare(task)


if __name__ == "__main__":
    main()
