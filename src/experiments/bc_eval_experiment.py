"""
BC Eval: pure evaluation of a fixed behaviour-cloning policy.
No learning — loads the R2BC checkpoint, runs deterministic rollouts,
and prints the mean episode reward.
"""

import torch
from pathlib import Path

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType, set_exploration_type

from src.experiments.base_marl_experiment import BaseMARLExperiment
from src.experiments.ibmarl.networks import R2bcPolicy


class BcEvalExperiment(BaseMARLExperiment):

    def __init__(self, config):
        super().__init__(config)

        bc_path = Path(config["r2bc_checkpoint_path"])
        self.bc_policy = R2bcPolicy(bc_path, self.env, self.device)

        self._bc_td_policy = self._wrap_bc_as_tensordict_policy()

    def _wrap_bc_as_tensordict_policy(self) -> TensorDictSequential:
        """Wrap R2bcPolicy.get_action into TensorDictModules so env.rollout works."""
        modules = []
        for group in self.env.group_map:
            def _forward(obs, _group=group):
                if obs.dim() == 2:
                    obs = obs.unsqueeze(0)
                    return self.bc_policy.get_action(_group, obs).squeeze(0)
                return self.bc_policy.get_action(_group, obs)

            modules.append(
                TensorDictModule(
                    _forward,
                    in_keys=[(group, "observation")],
                    out_keys=[(group, "action")],
                )
            )
        return TensorDictSequential(*modules)

    def train(self) -> str:
        n_episodes = 20
        horizon = self.config.get("horizon", 100)
        max_steps = horizon * (n_episodes + 5)

        print(f"\n{'='*60}")
        print(f"BC Policy Evaluation  ({n_episodes} episodes)")
        print(f"{'='*60}")

        self._bc_td_policy.eval()

        n_envs = max(int(self.env.batch_size[0]) if len(self.env.batch_size) else 1, 1)
        n_rollouts = max(1, -(-n_episodes // n_envs))  # ceil

        collected = {group: [] for group in self.env.group_map}
        with torch.no_grad():
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                for _ in range(n_rollouts):
                    out = self.env.rollout(horizon, policy=self._bc_td_policy)
                    for group in self.env.group_map:
                        ep_reward = out.get(("next", group, "episode_reward"))
                        done = self._agent_done_mask(out, group, ep_reward)
                        collected[group].append(ep_reward[done].float())

        lines = []
        for group in self.env.group_map:
            ep_rewards = (
                torch.cat(collected[group]) if collected[group] else torch.empty(0)
            )
            if ep_rewards.numel() == 0:
                mean_r = 0.0
                n_completed = 0
            else:
                n_completed = ep_rewards.shape[0]
                mean_r = ep_rewards.mean().item()

            std = ep_rewards.std().item() if ep_rewards.numel() > 1 else 0.0
            sem = std / (n_completed ** 0.5) if n_completed > 1 else 0.0

            # Logged so the IL reference line is a file on disk like every other
            # result, rather than a number that only exists in a console log.
            self.metrics_logger.log(
                iteration=0,
                group=group,
                eval_reward_mean=mean_r,
                eval_reward_std=round(std, 6),
                eval_reward_sem=round(sem, 6),
                n_agent_episodes=n_completed,
            )

            msg = (
                f"  {group:>20s}:  mean_reward = {mean_r:+.4f} +/- {sem:.4f} "
                f"(agent-episodes: {n_completed})"
            )
            print(msg)
            lines.append(msg)

        print(f"{'='*60}\n")
        return "BC eval complete.\n" + "\n".join(lines)

    def save_results(self):
        # Metrics only: there is no policy to checkpoint and no learning curve
        # to plot, so the base implementation does not apply.
        self.metrics_logger.save()
        print(f"Saved metrics to: {self.metrics_logger.path.resolve()}")

    def save_checkpoint(self):
        pass

    def render_policy(self):
        pass
