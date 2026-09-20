"""
RLfD (Reinforcement Learning from Demonstrations) experiment.

Controlled variant of MADDPG where the ONLY difference is replay sampling:
- A demonstration buffer is pre-loaded from disk and an online buffer is filled
  during interaction.
- Each training step samples a minibatch that is a mix of demo and online
  transitions, with the demo fraction linearly annealed from 50% to 0% over
  n_iters.

All other aspects (policy, critic, loss, data collection, optimiser steps)
are identical to MADDPG.
"""

from copy import deepcopy
from typing import Dict
import time

import torch

from tensordict import TensorDict, TensorDictBase

from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer

from src.util.demonstrations import load_demonstrations_into_buffer
from src.util.paths import resolve_path

from src.experiments.base_marl_experiment import BaseMARLExperiment
from src.experiments.maddpg import MaddpgExperiment


def _build_demo_and_online_buffers(cfg, env, device) -> tuple:
    """
    Build demo (pre-loaded) and online (empty) replay buffers per group.
    Same storage/sampler/batch_size as MADDPG.
    """
    demonstration_path = resolve_path(cfg["demonstrations_path"])
    memory_size = cfg["memory_size"]
    train_batch_size = cfg["training"]["train_batch_size"]

    demo_buffers: Dict[str, ReplayBuffer] = {}
    online_buffers: Dict[str, ReplayBuffer] = {}

    for group, _agents in env.group_map.items():
        for name, buffers in (("demo", demo_buffers), ("online", online_buffers)):
            rb = ReplayBuffer(
                storage=LazyMemmapStorage(memory_size),
                sampler=RandomSampler(),
                batch_size=train_batch_size,
            )
            if device.type != "cpu":
                rb.append_transform(lambda td: td.to(device))
            buffers[group] = rb

        load_demonstrations_into_buffer(demonstration_path, group, demo_buffers[group], log_prefix="[RLfD]")

    return demo_buffers, online_buffers


def _concat_minibatches(
    td1: TensorDictBase, td2: TensorDictBase
) -> TensorDictBase:
    """Concatenate two TensorDict minibatches along batch dimension."""
    total_batch = td1.batch_size[0] + td2.batch_size[0]
    device = td1.device
    out = TensorDict({}, batch_size=[total_batch], device=device)

    for key in td1.keys(True, True):
        v1 = td1.get(key)
        v2 = td2.get(key)
        if v1 is None or v2 is None:
            continue
        out.set(key, torch.cat([v1, v2], dim=0))

    return out


class RlfdExperiment(MaddpgExperiment):
    """
    MADDPG with demonstration oversampling: same policy, critic, loss, and
    training step as MADDPG; only the replay sampling differs (demo buffer +
    online buffer, with linear annealing of demo fraction from 50% to 0%).
    """

    def __init__(self, config):
        super().__init__(config)

        demo_buffers, online_buffers = _build_demo_and_online_buffers(
            config, self.env, self.device
        )
        self.demo_replay_buffers = demo_buffers
        self.replay_buffers = online_buffers

    def _resume_buffers(self):
        buffers = super()._resume_buffers()
        # The demo buffer is rebuilt deterministically from disk at construction,
        # so it does not need dumping -- only the online buffer carries run state.
        return buffers

    def train(self):
        print("Training RLfD Experiment (MADDPG + demo sampling)...")

        from tqdm import tqdm

        start_iteration, counters = self.begin_training(
            {
                "total_frames": 0,
                "total_episodes": 0,
                "total_train_steps": 0,
                "elapsed": 0.0,
            }
        )

        pbar = tqdm(
            total=self.config.get("n_iters"),
            initial=start_iteration,
            desc=", ".join(
                [f"episode_reward_mean_{group}=0" for group in self.env.group_map.keys()]
            ),
        )

        train_group_map = deepcopy(self.env.group_map)
        n_iters = self.config.get("n_iters")
        total_iters = max(1, n_iters - 1)
        train_batch_size = self.config.get("training").get("train_batch_size")

        start_time = time.time()
        elapsed_offset = counters["elapsed"]
        total_frames = counters["total_frames"]
        total_episodes = counters["total_episodes"]
        total_train_steps = counters["total_train_steps"]

        for offset, batch in enumerate(self.collector):
            iteration = start_iteration + offset
            current_frames = batch.numel()
            total_frames += current_frames
            batch = self.process_batch(batch)

            actor_losses = {group: [] for group in self.env.group_map.keys()}
            critic_losses = {group: [] for group in self.env.group_map.keys()}

            for group in train_group_map.keys():
                group_batch = batch.exclude(
                    *[
                        key
                        for _group in self.env.group_map.keys()
                        if _group != group
                        for key in [_group, ("next", _group)]
                    ]
                )
                group_batch = group_batch.reshape(-1)

                self.replay_buffers[group].extend(group_batch)

                progress = min(1.0, float(iteration) / float(total_iters))
                demo_frac = 0.5 * (1.0 - progress)
                demo_batch_size = int(round(train_batch_size * demo_frac))
                if demo_batch_size >= train_batch_size:
                    demo_batch_size = train_batch_size - 1
                demo_batch_size = max(0, demo_batch_size)
                online_batch_size = train_batch_size - demo_batch_size

                for _ in range(self.config.get("training").get("n_optimiser_steps")):
                    if demo_batch_size > 0 and online_batch_size > 0:
                        demo_mb = self.demo_replay_buffers[group].sample(
                            batch_size=demo_batch_size
                        )
                        online_mb = self.replay_buffers[group].sample(
                            batch_size=online_batch_size
                        )
                        minibatch = _concat_minibatches(demo_mb, online_mb)
                    elif demo_batch_size > 0:
                        minibatch = self.demo_replay_buffers[group].sample(
                            batch_size=demo_batch_size
                        )
                    else:
                        minibatch = self.replay_buffers[group].sample(
                            batch_size=online_batch_size
                        )

                    loss_vals = self.losses[group](minibatch)

                    actor_losses[group].append(loss_vals["loss_actor"].item())
                    critic_losses[group].append(loss_vals["loss_value"].item())

                    for loss_name in ["loss_actor", "loss_value"]:
                        loss = loss_vals[loss_name]
                        optimiser = self.optimisers[group][loss_name]
                        loss.backward()
                        params = optimiser.param_groups[0]["params"]
                        torch.nn.utils.clip_grad_norm_(
                            params,
                            self.config.get("training").get("max_grad_norm"),
                        )
                        optimiser.step()
                        optimiser.zero_grad()

                    self.target_updaters[group].step()
                    total_train_steps += 1

                self.exploration_policies[group][-1].step(current_frames)

            # --- Dedicated evaluation (thinned by eval_interval; None rows are
            # skipped by MetricsLogger.log and dropped by the analysis scripts) ---
            eval_means = None
            if self.should_evaluate(iteration):
                eval_means = self.evaluate_at_iteration(iteration, n_episodes=20)

            # --- Metrics ---
            elapsed = elapsed_offset + (time.time() - start_time)
            speed = total_frames / max(elapsed, 1e-6)

            done_global = batch.get(("next", "done"))
            episodes_this_iter = int(done_global.sum().item())
            total_episodes += episodes_this_iter

            for group in self.env.group_map.keys():
                done = batch.get(("next", group, "done"))
                ep_rewards = batch.get(("next", group, "episode_reward"))[done]
                episode_reward_mean = ep_rewards.mean().item() if ep_rewards.numel() > 0 else 0.0

                n_opt = max(len(actor_losses[group]), 1)

                self.metrics_logger.log(
                    iteration=iteration,
                    group=group,
                    elapsed_time=round(elapsed, 2),
                    episode=total_episodes,
                    step=total_frames,
                    train_step=total_train_steps,
                    speed_fps=round(speed, 2),
                    episode_reward_mean=episode_reward_mean,
                    eval_reward_mean=eval_means[group] if eval_means is not None else None,
                    actor_loss=round(sum(actor_losses[group]) / n_opt, 6),
                    critic_loss=round(sum(critic_losses[group]) / n_opt, 6),
                    replay_size=len(self.replay_buffers[group]),
                    demo_fraction=round(demo_frac, 4),
                )

            if iteration % 10 == 0:
                self.metrics_logger.save()

            if (iteration + 1) % self.resume_interval == 0:
                self.metrics_logger.save()
                self.save_resume(
                    iteration,
                    {
                        "total_frames": total_frames,
                        "total_episodes": total_episodes,
                        "total_train_steps": total_train_steps,
                        "elapsed": elapsed,
                    },
                )

            pbar.set_description(
                ", ".join(
                    [
                        f"episode_reward_mean_{group} = "
                        f"{self.metrics_logger.get_values('episode_reward_mean', group)[-1]}"
                        for group in self.env.group_map.keys()
                    ]
                ),
                refresh=False,
            )
            pbar.update()

        self.metrics_logger.save()
        self.clear_resume()

        first_group = list(self.env.group_map.keys())[0]
        recent = self.metrics_logger.get_values("episode_reward_mean", first_group)[-10:]
        return (
            f"RLfD training complete. Environment: {self.config['scenario_name']}, "
            f"Experiment Type: {self.experiment_type}, Seed: {self.seed}.\n\n"
            f"Results: {recent}"
        )
