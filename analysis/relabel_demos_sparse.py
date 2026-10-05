"""
Relabel balance's dense R2BC recording to the gt-in-radius sparse schema.

teachers/balance/demonstrations.pt was recorded with the raw VMAS reward (100 x
the per-step decrease in package-goal distance, -10 on floor contact) at a fixed
300-step horizon, 36 episodes, 4-env interleaved rows (index = t*4 + e).  The
online task (config/environments/balance.yaml, sparse_rewards True) pays that
native reward only while the package is within ``gt_radius`` of the goal and a
flat -1 per step otherwise (transforms/balance_sparse.py), so the demonstrations
the demo-based learners pre-load must carry the same labels or the replay buffer
mixes two reward scales.

The relabel is row-wise and exact: the transform decides on the POST-step
observation, i.e. ``next_obs[:, 0, 8:10]`` (``package.pos - goal.pos``, the same
for every agent), so

    reward[t] = native[t]  if ||next_obs[t, 0, 8:10]|| < radius  else  -1

``dones`` and ``terminated`` are written all-False: the schema has no terminal
and every episode runs the full horizon (scenarios/balance_fixed_horizon.py).
The recorder's mid-episode on-goal flag, which the old file carried as
``dones``, is kept under ``on_goal`` for provenance only -- the loader
(src/util/demonstrations.py) reads ``dones``/``terminated`` and ignores it.

The radius defaults to the ``gt_radius`` in config/environments/balance.yaml so
the file and the online env cannot drift apart silently; ``meta.radius``
records what was used.  Unlike relabel_demos_binary.py nothing is zeroed or
cut, so the layout (env-interleaved or episode-contiguous) does not matter for
the labels; the episode view is only needed for the statistics, which exist to
choose the radius (see analysis/balance_teacher_audit.py for the policy side):

    python -m analysis.relabel_demos_sparse balance --radii 0.3,0.5,0.75,1.05 --stats-only
    python -m analysis.relabel_demos_sparse balance                 # radius from the yaml
    python -m analysis.relabel_demos_sparse balance --radius 0.5 --src teachers/balance_18demo/demonstrations.pt \
        --dst teachers/balance_18demo/demonstrations_sparse.pt

Writes teachers/<task>/demonstrations_sparse.pt by default (never the source).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from analysis.relabel_demos_binary import infer_env_count, to_episodes
from src.util.paths import PROJECT_ROOT

TASKS = {
    "balance": dict(
        horizon=300,
        rel_slice=slice(8, 10),      # package.pos - goal.pos in vmas 1.5.2's balance observation
        out_penalty=-1.0,
        env_yaml="config/environments/balance.yaml",
        fall_reward=-10.0,           # vmas balance fall_reward; a row below -5 is a floor contact
    ),
}

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def config_radius(task: str) -> float:
    with open(PROJECT_ROOT / TASKS[task]["env_yaml"]) as f:
        cfg = yaml.safe_load(f) or {}
    if "gt_radius" not in cfg:
        raise SystemExit(f"{TASKS[task]['env_yaml']} has no gt_radius; pass --radius")
    return float(cfg["gt_radius"])


def load_recording(src: Path) -> dict[str, np.ndarray]:
    d = torch.load(src, map_location="cpu", weights_only=False)
    out = {
        "obs": np.asarray(d["obs"], dtype=np.float32),
        "act": np.asarray(d["act"], dtype=np.float32),
        "rewards": np.asarray(d["rewards"], dtype=np.float32),
        "next_obs": np.asarray(d["next_obs"], dtype=np.float32),
        "dones": np.asarray(d["dones"], dtype=bool),
    }
    if out["rewards"].ndim == 3:  # [N, A, 1] -> [N, A]
        out["rewards"] = out["rewards"][..., 0]
    n, a = out["obs"].shape[:2]
    assert out["rewards"].shape == (n, a), out["rewards"].shape
    assert out["next_obs"].shape == out["obs"].shape
    return out


def goal_distance(next_obs: np.ndarray, rel_slice: slice) -> np.ndarray:
    """||package - goal|| per row, from agent 0's post-step observation."""
    return np.linalg.norm(next_obs[:, 0, rel_slice], axis=-1)


def relabel_rows(rewards: np.ndarray, dist: np.ndarray, radius: float, out_penalty: float):
    inside = dist < radius                                   # [N]
    new = np.where(inside[:, None], rewards, np.float32(out_penalty)).astype(np.float32)
    return new, inside


def episode_view(rec: dict[str, np.ndarray], horizon: int) -> tuple[dict[str, np.ndarray], int, float]:
    """Episode-major arrays plus the inferred env interleave (E=1 for contiguous files)."""
    n_envs, cont = infer_env_count(rec["obs"], rec["next_obs"], horizon)
    ep = {k: to_episodes(v, n_envs, horizon) for k, v in rec.items()}
    return ep, n_envs, cont


def radius_stats(ep: dict[str, np.ndarray], spec: dict, radius: float) -> dict:
    """Per-radius numbers the radius choice is made on (team-level, agent 0)."""
    H = spec["horizon"]
    dist = np.linalg.norm(ep["next_obs"][:, :, 0, spec["rel_slice"]], axis=-1)   # (n_ep, H)
    native = ep["rewards"][:, :, 0]                                                # (n_ep, H)
    inside = dist < radius
    ever = inside.any(1)
    first = inside.argmax(1)[ever] + 1                                             # 1-based step
    ret = np.where(inside, native, spec["out_penalty"]).sum(1)
    return dict(
        radius=float(radius),
        episodes_ever_inside=int(ever.sum()),
        frac_rows_inside=round(float(inside.mean()), 4),
        first_entry_step_q25_50_75=[int(x) for x in np.percentile(first, [25, 50, 75])] if ever.any() else None,
        steps_inside_mean=round(float(inside.sum(1).mean()), 1),
        return_mean=round(float(ret.mean()), 1),
        return_min=round(float(ret.min()), 1),
        return_max=round(float(ret.max()), 1),
        horizon=H,
    )


def shape_stats(ep: dict[str, np.ndarray], spec: dict) -> dict:
    """Radius-independent description of the recording."""
    dist = np.linalg.norm(ep["next_obs"][:, :, 0, spec["rel_slice"]], axis=-1)
    dist0 = np.linalg.norm(ep["obs"][:, 0, 0, spec["rel_slice"]], axis=-1)
    native = ep["rewards"][:, :, 0]
    fall_rows = native < spec["fall_reward"] / 2
    return dict(
        n_episodes=int(dist.shape[0]),
        initial_dist_min_max=[round(float(dist0.min()), 3), round(float(dist0.max()), 3)],
        min_dist_quantiles={str(q): round(float(np.quantile(dist.min(1), q)), 3) for q in QUANTILES},
        final_dist_mean=round(float(dist[:, -1].mean()), 3),
        native_return_mean=round(float(native.sum(1).mean()), 1),
        fall_rows=int(fall_rows.sum()),
        episodes_with_fall=int(fall_rows.any(1).sum()),
        on_goal_flag_rows=int(ep["dones"].sum()),
    )


def print_table(rows: list[dict]) -> None:
    cols = ["radius", "episodes_ever_inside", "frac_rows_inside",
            "first_entry_step_q25_50_75", "steps_inside_mean", "return_mean", "return_min", "return_max"]
    print("  ".join(f"{c:>26}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>26}" for c in cols))


def relabel(task: str, src: Path, dst: Path | None, radius: float, radii: list[float]) -> dict | None:
    spec = TASKS[task]
    rec = load_recording(src)
    n_rows = rec["obs"].shape[0]
    assert n_rows % spec["horizon"] == 0, (n_rows, spec["horizon"])
    ep, n_envs, cont = episode_view(rec, spec["horizon"])

    shape = shape_stats(ep, spec)
    print(f"{src.relative_to(PROJECT_ROOT) if src.is_relative_to(PROJECT_ROOT) else src}: "
          f"{n_rows} rows, {shape['n_episodes']} episodes of {spec['horizon']}, "
          f"env interleave {n_envs} (continuity {cont:.4f})")
    for k, v in shape.items():
        print(f"  {k}: {v}")
    print_table([radius_stats(ep, spec, r) for r in sorted(set(radii) | {radius})])

    if dst is None:
        return None

    dist = goal_distance(rec["next_obs"], spec["rel_slice"])
    new_rew, inside = relabel_rows(rec["rewards"], dist, radius, spec["out_penalty"])
    assert np.all(new_rew[~inside] == spec["out_penalty"]), "relabel mask mismatch (outside rows)"
    assert np.array_equal(new_rew[inside], rec["rewards"][inside]), "relabel mask mismatch (inside rows)"

    meta = dict(
        source=str(src.relative_to(PROJECT_ROOT)) if src.is_relative_to(PROJECT_ROOT) else str(src),
        task=task, radius=float(radius), out_penalty=spec["out_penalty"], rel_slice=[spec["rel_slice"].start, spec["rel_slice"].stop],
        horizon=spec["horizon"], n_envs=int(n_envs), continuity=round(float(cont), 4),
        n_transitions=int(n_rows),
        schema=("native VMAS reward where ||next_obs[:,0,8:10]|| < radius, else out_penalty; "
                "fixed horizon, no terminal: dones/terminated all-False; on_goal = recorder's flag"),
        **shape,
        at_radius=radius_stats(ep, spec, radius),
        grid=[radius_stats(ep, spec, r) for r in sorted(set(radii) | {radius})],
    )
    out = {
        "obs": torch.from_numpy(rec["obs"]),
        "act": torch.from_numpy(rec["act"]),
        "rewards": torch.from_numpy(new_rew),
        "next_obs": torch.from_numpy(rec["next_obs"]),
        "dones": torch.zeros(n_rows, dtype=torch.bool),
        "terminated": torch.zeros(n_rows, dtype=torch.bool),
        "on_goal": torch.from_numpy(rec["dones"]),
        "meta": meta,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"wrote {dst}  (radius {radius:g}: {meta['at_radius']['frac_rows_inside']:.1%} of rows inside, "
          f"{meta['at_radius']['episodes_ever_inside']}/{shape['n_episodes']} episodes ever inside)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--radius", type=float, default=None,
                    help="basin radius; default: gt_radius from the task's environment yaml")
    ap.add_argument("--radii", default="0.3,0.5,0.75,1.05",
                    help="comma-separated grid for the statistics table")
    ap.add_argument("--stats-only", action="store_true", help="print the tables, write nothing")
    ap.add_argument("--src", type=Path, default=None, help="default teachers/<task>/demonstrations.pt")
    ap.add_argument("--dst", type=Path, default=None, help="default teachers/<task>/demonstrations_sparse.pt")
    args = ap.parse_args()

    radius = args.radius if args.radius is not None else config_radius(args.task)
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    src = (args.src or PROJECT_ROOT / "teachers" / args.task / "demonstrations.pt").resolve()
    dst = None if args.stats_only else (args.dst or PROJECT_ROOT / "teachers" / args.task / "demonstrations_sparse.pt").resolve()
    if dst is not None and src == dst:
        raise SystemExit("refusing to overwrite the source recording")
    relabel(args.task, src, dst, radius, radii)


if __name__ == "__main__":
    main()
