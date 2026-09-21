"""
BC Eval: pure evaluation of a fixed behaviour-cloning policy.
No learning — loads the R2BC checkpoint, runs deterministic rollouts,
and prints the mean episode reward.
"""

import torch

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType, set_exploration_type

from src.experiments.base_marl_experiment import BaseMARLExperiment
from src.experiments.ibmarl.networks import R2bcPolicy
from src.util.paths import resolve_path


class BcEvalExperiment(BaseMARLExperiment):

    def __init__(self, config):
        super().__init__(config)

        bc_path = resolve_path(config["r2bc_checkpoint_path"])
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

        # The dedicated eval env (BaseMARLExperiment.eval_env): same protocol,
        # seed and `eval_episodes` batch as every other variant's evaluation.
        env = self.eval_env
        n_envs = max(int(env.batch_size[0]) if len(env.batch_size) else 1, 1)
        n_rollouts = max(1, -(-n_episodes // n_envs))  # ceil

        collected = {group: [] for group in self.env.group_map}
        with torch.no_grad():
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                for _ in range(n_rollouts):
                    # See BaseMARLExperiment.evaluate: the default stops at the
                    # first sub-env to finish and only ended episodes are
                    # counted, which discards the slower ones.  Barely moves the
                    # BC number (it rarely reaches the goal) but keeps this on
                    # the same measurement protocol as every other variant.
                    out = env.rollout(
                        horizon,
                        policy=self._bc_td_policy,
                        break_when_any_done=False,
                    )
                    for group in env.group_map:
                        collected[group].append(
                            self._completed_episode_returns(out, group)
                        )

        lines = []
        for group, agents in env.group_map.items():
            ep_rewards = (
                torch.cat(collected[group]) if collected[group] else torch.empty(0)
            )
            if ep_rewards.numel() == 0:
                mean_r = 0.0
                n_completed = 0
            else:
                n_completed = ep_rewards.shape[0]
                mean_r = ep_rewards.mean().item()

            # Returns arrive per agent-episode.  Where the reward is team-shared
            # (buzz_wire pays both agents together) the n_agents values of one
            # episode are copies, so the independent sample is the EPISODE:
            # dividing by agent-episodes understated the SEM by sqrt(n_agents).
            # Detected from the data rather than assumed, so navigation's
            # per-agent reward keeps its per-agent-episode SEM.
            n_agents = max(len(agents), 1)
            n_episodes_done = n_completed // n_agents
            per_ep = ep_rewards.view(n_episodes_done, n_agents) if n_completed and n_completed % n_agents == 0 else None
            team_shared = bool(per_ep is not None and per_ep.shape[0] > 0
                               and torch.equal(per_ep.max(dim=1).values, per_ep.min(dim=1).values))
            sample = per_ep[:, 0] if team_shared else ep_rewards
            n_indep = int(sample.numel())
            std = sample.std().item() if n_indep > 1 else 0.0
            sem = std / (n_indep ** 0.5) if n_indep > 1 else 0.0

            # Logged so the IL reference line is a file on disk like every other
            # result, rather than a number that only exists in a console log.
            self.metrics_logger.log(
                iteration=0,
                group=group,
                eval_reward_mean=mean_r,
                eval_reward_std=round(std, 6),
                eval_reward_sem=round(sem, 6),
                n_agent_episodes=n_completed,
                n_episodes=n_episodes_done,
                team_shared_reward=team_shared,
            )

            msg = (
                f"  {group:>20s}:  mean_reward = {mean_r:+.4f} +/- {sem:.4f} "
                f"(episodes: {n_episodes_done}, agent-episodes: {n_completed}, "
                f"SEM over {'episodes' if team_shared else 'agent-episodes'})"
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
