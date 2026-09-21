"""
Relabel recorded R2BC demonstrations to the binary terminal-reward schema.

The teacher recordings in teachers/<task>/demonstrations.pt were made under the
legacy sparse schema (-1 per step off-goal, fixed horizon, done() suppressed).
IBMARL's navigation and buzz_wire tasks now pay +1 once, on the first step the
success predicate holds, and terminate there.  This script derives that
labelling from the recordings without re-simulating:

  navigation  per agent: agent i succeeds on the first step ||next_obs[i, 4:6]||
              < 0.05 (the goal radius) -- the predicate TerminalSuccess-
              NavigationScenario pays on.  Team terminal = every agent has
              succeeded.
  buzz_wire   team: the ball is inside the 0.1 basin exactly when the recorded
              reward != -1 (the legacy inside branch is pos_rew, and a one-step
              distance change of exactly 1.0 is impossible).  The ball position
              is not in the observation, so this is the only way to recover it.
              Online, wall contact also ends the episode with 0; the recording
              cannot see contacts, so episodes are cut at the basin regardless
              (~8% of the teacher's successes had a prior touch -- a small,
              documented optimism, see teachers/README.md).

Per episode: every reward is zeroed; each agent gets +1 at its own first-success
step; the episode is cut at the team terminal, whose transition gets done=True
and terminated=True.  Episodes that never terminate keep all H transitions with
terminated=False (a time-limit end -- the loaders bootstrap through it).

Layout: R2BC's recorder appends one row per (step, sub-env), so the flat arrays
are time-major / env-minor (index = t*E + e), NOT episode-contiguous.  E is
inferred from within-episode observation continuity (next_obs[t] == obs[t+1])
and asserted; a naive (N//H, H, ...) reshape interleaves envs and yields garbage.

Writes teachers/<task>/demonstrations_binary.pt (never touches the source).

    python -m analysis.relabel_demos_binary navigation
    python -m analysis.relabel_demos_binary buzz_wire
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.util.paths import PROJECT_ROOT

TASKS = {
    # horizon: R2BC/config/environments/<task>.yaml; predicate: see module docstring
    "navigation": dict(horizon=100, predicate="per_agent_goal", threshold=0.05),
    "buzz_wire": dict(horizon=200, predicate="team_reward_mask", threshold=0.1),
}


def to_episodes(flat: np.ndarray, n_envs: int, horizon: int) -> np.ndarray:
    """(N, ...) time-major/env-minor rows -> (n_episodes, H, ...), env-major."""
    n = flat.shape[0]
    a = flat.reshape(n // n_envs, n_envs, *flat.shape[1:])  # [t, e, ...]
    a = np.moveaxis(a, 1, 0)                                # [e, t, ...]
    return a.reshape(n_envs * (n // n_envs // horizon), horizon, *flat.shape[1:])


def infer_env_count(obs: np.ndarray, next_obs: np.ndarray, horizon: int) -> tuple[int, float]:
    n = obs.shape[0]
    best = None
    for n_envs in range(1, 65):
        if n % n_envs or (n // n_envs) % horizon:
            continue
        o = to_episodes(obs, n_envs, horizon)
        nx = to_episodes(next_obs, n_envs, horizon)
        # pos + vel only: lidar/goal channels are the noisiest and add nothing here
        cont = float(np.isclose(nx[:, :-1, :, :4], o[:, 1:, :, :4], atol=1e-3).mean())
        if best is None or cont > best[1]:
            best = (n_envs, cont)
    n_envs, cont = best
    if cont < 0.98:
        raise RuntimeError(
            f"no env count gives contiguous episodes (best E={n_envs}, continuity {cont:.3f}); "
            "the recorder layout assumption in this script no longer holds"
        )
    return n_envs, cont


def relabel(task: str, src: Path, dst: Path) -> dict:
    spec = TASKS[task]
    H, thr = spec["horizon"], spec["threshold"]
    d = torch.load(src, map_location="cpu", weights_only=False)
    obs, act = np.array(d["obs"]), np.array(d["act"])
    rew, next_obs = np.array(d["rewards"]), np.array(d["next_obs"])
    n_rows, n_agents = obs.shape[0], obs.shape[1]
    assert n_rows % H == 0, (n_rows, H)

    n_envs, cont = infer_env_count(obs, next_obs, H)
    ep = {k: to_episodes(v, n_envs, H) for k, v in
          dict(obs=obs, act=act, rewards=rew, next_obs=next_obs).items()}
    n_ep = ep["obs"].shape[0]

    if spec["predicate"] == "per_agent_goal":
        dist = np.linalg.norm(ep["next_obs"][..., 4:6], axis=-1)        # (n_ep, H, A)
        hit = dist < thr                                                # agent i on goal at step t
    else:  # team_reward_mask
        inside = ep["rewards"][..., 0] != -1.0                          # (n_ep, H)
        hit = np.repeat(inside[..., None], n_agents, axis=-1)           # shared: both agents at once
    paid_by = np.cumsum(hit, axis=1) > 0                                # paid on or before step t
    team_done = paid_by.all(-1)                                         # (n_ep, H)

    ever = hit.any(1)                                                   # (n_ep, A)
    first = hit.argmax(1)                                               # (n_ep, A) first-success step
    terminates = team_done.any(1)                                       # (n_ep,)
    t_term = np.where(terminates, team_done.argmax(1), H - 1)           # last kept step
    lengths = t_term + 1

    new_rew = np.zeros_like(ep["rewards"], dtype=np.float32)
    for e in range(n_ep):
        for a in range(n_agents):
            if ever[e, a]:
                assert first[e, a] <= t_term[e]
                new_rew[e, first[e, a], a] = 1.0
    dones = np.zeros((n_ep, H), dtype=bool)
    terminated = np.zeros((n_ep, H), dtype=bool)
    dones[np.arange(n_ep), t_term] = True
    terminated[np.arange(n_ep), t_term] = terminates

    keep = np.arange(H)[None, :] < lengths[:, None]                     # (n_ep, H)
    out = {
        "obs": ep["obs"][keep], "act": ep["act"][keep],
        "rewards": new_rew[keep], "next_obs": ep["next_obs"][keep],
        "dones": dones[keep], "terminated": terminated[keep],
    }
    meta = dict(
        source=str(src.relative_to(PROJECT_ROOT)), task=task, horizon=H, n_envs=n_envs,
        continuity=round(cont, 4), predicate=spec["predicate"], threshold=thr,
        n_episodes=int(n_ep), n_terminated=int(terminates.sum()),
        n_agent_successes=int(ever.sum()), n_agent_episodes=int(ever.size),
        n_transitions_in=int(n_rows), n_transitions_out=int(keep.sum()),
        first_success_steps=sorted(int(t) + 1 for t in t_term[terminates]),
        schema="reward 1 once per agent at its first success, else 0; episode cut at the team "
               "terminal (done=terminated=True); untermininated episodes end done=True, terminated=False",
    )
    out["meta"] = meta
    torch.save(out, dst)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--src", type=Path, default=None, help="default teachers/<task>/demonstrations.pt")
    ap.add_argument("--dst", type=Path, default=None, help="default teachers/<task>/demonstrations_binary.pt")
    args = ap.parse_args()
    src = args.src or PROJECT_ROOT / "teachers" / args.task / "demonstrations.pt"
    dst = args.dst or PROJECT_ROOT / "teachers" / args.task / "demonstrations_binary.pt"
    if src.resolve() == dst.resolve():
        raise SystemExit("refusing to overwrite the source recording")
    meta = relabel(args.task, src, dst)
    print(f"wrote {dst}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
