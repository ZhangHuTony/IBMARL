"""
Success rate of candidate teacher checkpoints under the task's actual reward
schema, measured on the dedicated eval env at a chosen episode count.

    python -m analysis.teacher_success_curve --scenario buzz_wire --episodes 400 \
        mf4.0=/path/to/run_a/policy_checkpoint.pth mf2.0=/path/to/run_b/policy_checkpoint.pth

Exists because R2BC's own metrics.csv scores a checkpoint under the legacy
-1/step schema on 50 fixed episodes, which ranks checkpoints differently from
the binary terminal schema (legacy scoring ignores wall contact; see
config/experiments/ibmarl_buzz_wire.yaml) -- and because bc_eval's single run is
one draw.  This rolls each checkpoint through BcEvalExperiment's own policy
wrapper on eval_env (so `eval_episodes` sizes the batch), for as many rollouts
as --episodes needs, and counts EPISODES: on a team-shared reward the n_agents
returns of one episode are copies and must not be counted as independent
samples (asserted, not assumed).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math

import torch
from torchrl.envs import ExplorationType, set_exploration_type

from analysis.paper_run import build_cfg
from src.experiments.bc_eval_experiment import BcEvalExperiment


def success_rate(exp: BcEvalExperiment, episodes: int, seed: int) -> tuple[float, float, int]:
    env = exp.eval_env
    n_envs = int(env.batch_size[0])
    n_rollouts = max(1, -(-episodes // n_envs))
    horizon = exp.config["horizon"]
    exp.reseed_eval_env(seed)
    per_ep: list[float] = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(n_rollouts):
            out = env.rollout(horizon, policy=exp._bc_td_policy, break_when_any_done=False)
            for group, agents in env.group_map.items():
                vals = exp._completed_episode_returns(out, group).view(-1, len(agents))
                assert torch.equal(vals.max(dim=1).values, vals.min(dim=1).values), \
                    "agents disagree within an episode: reward is not team-shared, count agent-episodes instead"
                per_ep += vals[:, 0].float().tolist()
    k = len(per_ep)
    p = sum(per_ep) / k
    return p, math.sqrt(p * (1 - p) / k), k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidates", nargs="+", help="name=path/to/policy_checkpoint.pth")
    ap.add_argument("--scenario", default="buzz_wire")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0, help="eval-env seed, shared by every candidate")
    args = ap.parse_args()

    print(f"{'candidate':<22} {'success':>8} {'+/-':>6} {'episodes':>9}")
    print("-" * 50)
    for spec in args.candidates:
        name, path = spec.split("=", 1)
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = build_cfg("bc_eval", args.scenario, 0, None, tag="_teacher_curve")
            cfg["r2bc_checkpoint_path"] = path
            exp = BcEvalExperiment(cfg)
            p, sem, k = success_rate(exp, args.episodes, args.seed)
        print(f"{name:<22} {p:>8.3f} {sem:>6.3f} {k:>9}")


if __name__ == "__main__":
    main()
