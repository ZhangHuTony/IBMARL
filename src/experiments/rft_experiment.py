"""
RFT (Reinforcement Fine-Tuning) experiment.

This is a controlled variant of MADDPG where the ONLY difference is that,
during RL training, the actor loss is augmented with a behaviour cloning (BC)
regularizer that penalizes deviation from demonstration actions early in
training:

    loss_actor_rft = loss_actor_maddpg + alpha * lambda(t) * L_BC

where:
- alpha is a scalar hyperparameter (config, default 0.1),
- lambda(t) linearly anneals from 1.0 at iteration 0 to 0.0 after a
  predetermined number of iterations,
- L_BC is the mean-squared error between the policy actions and demonstration
  actions on a separate demonstration replay buffer.

All other aspects (policy / critic architectures, data collection, replay
buffer for RL data, and DDPG loss configuration) are identical to MADDPG.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict
import time

import numpy as np
import torch
import torch.nn.functional as F

from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer

from tqdm import tqdm

from src.experiments.maddpg import MaddpgExperiment


def _load_demonstrations_into_buffer(
    demonstration_path: Path, group: str, replay_buffer: ReplayBuffer
) -> None:
    """
    Load demonstrations into the given replay buffer, matching the
    [Batch, Agents, *] structure of the live collector (same as MADDPG batches).
    """
    if not demonstration_path or not demonstration_path.exists():
        raise FileNotFoundError(f"Demonstrations not found at {demonstration_path}")

    print(f"[RFT] Loading demonstrations for group '{group}' from {demonstration_path}")
    demo_data = torch.load(demonstration_path, map_location="cpu", weights_only=False)

    obs = torch.tensor(np.array(demo_data["obs"]), dtype=torch.float32)
    act = torch.tensor(np.array(demo_data["act"]), dtype=torch.float32)
    rew = torch.tensor(np.array(demo_data["rewards"]), dtype=torch.float32)

    if rew.ndim == 2:
        rew = rew.unsqueeze(-1)

    total_elements = obs.shape[0]
    n_agents = obs.shape[1]

    agents_data = TensorDict(
        {
            "observation": obs,
            "action": act,
            "episode_reward": rew,
        },
        batch_size=[total_elements, n_agents],
    )

    next_agents_data = agents_data.clone()
    next_agents_data.set(
        "done",
        torch.zeros((total_elements, n_agents, 1), dtype=torch.bool),
    )
    next_agents_data.set(
        "terminated",
        torch.zeros((total_elements, n_agents, 1), dtype=torch.bool),
    )
    next_agents_data.set("reward", rew.clone())

    next_td = TensorDict(
        {
            group: next_agents_data,
            "done": torch.zeros((total_elements, 1), dtype=torch.bool),
            "terminated": torch.zeros((total_elements, 1), dtype=torch.bool),
        },
        batch_size=[total_elements],
    )

    td = TensorDict(
        {
            group: agents_data,
            "next": next_td,
            "done": torch.zeros((total_elements, 1), dtype=torch.bool),
            "terminated": torch.zeros((total_elements, 1), dtype=torch.bool),
        },
        batch_size=[total_elements],
    )

    replay_buffer.extend(td)
    print(
        f"[RFT] Loaded {total_elements} demonstration transitions into {group} buffer."
    )


def _build_demo_buffers(cfg, env, device) -> Dict[str, ReplayBuffer]:
    """
    Build demonstration replay buffers per group.
    Same storage/sampler/batch_size as MADDPG, but used ONLY for the BC loss.
    """
    demonstration_path = Path(cfg["demonstrations_path"])
    memory_size = cfg["memory_size"]
    train_batch_size = cfg["training"]["train_batch_size"]

    demo_buffers: Dict[str, ReplayBuffer] = {}

    for group, _agents in env.group_map.items():
        rb = ReplayBuffer(
            storage=LazyMemmapStorage(memory_size),
            sampler=RandomSampler(),
            batch_size=train_batch_size,
        )
        if device.type != "cpu":
            rb.append_transform(lambda td: td.to(device))
        demo_buffers[group] = rb

        _load_demonstrations_into_buffer(demonstration_path, group, demo_buffers[group])

    return demo_buffers


class RftExperiment(MaddpgExperiment):
    """
    MADDPG with a behaviour-cloning regularizer on the actor.
    The BC loss is computed from a separate demonstration replay buffer and
    linearly annealed to zero over a configurable number of iterations.
    """

    def __init__(self, config):
        super().__init__(config)

        self.demo_replay_buffers = _build_demo_buffers(
            config, self.env, self.device
        )

        rft_cfg = config.get("rft", {})
        self._bc_alpha = float(rft_cfg.get("bc_alpha", 0.1))
        self._bc_anneal_n_iters = int(
            rft_cfg.get("bc_anneal_n_iters", config.get("n_iters"))
        )
        self._bc_batch_size = int(
            rft_cfg.get(
                "bc_batch_size", config.get("training", {}).get("train_batch_size", 1024)
            )
        )

    def _bc_weight(self, iteration: int) -> float:
        if self._bc_alpha <= 0.0 or self._bc_anneal_n_iters <= 0:
            return 0.0

        t = min(iteration, self._bc_anneal_n_iters)
        progress = float(t) / float(max(1, self._bc_anneal_n_iters))
        lam = max(0.0, 1.0 - progress)
        return self._bc_alpha * lam

    def _compute_bc_loss_for_group(self, group: str) -> torch.Tensor:
        demo_mb = self.demo_replay_buffers[group].sample(
            batch_size=self._bc_batch_size
        )

        target_actions = demo_mb.get((group, "action")).clone()

        policy_td = self.policies[group](demo_mb)
        pred_actions = policy_td.get((group, "action"))

        return F.mse_loss(pred_actions, target_actions)

    def train(self):
        print("Training RFT Experiment (MADDPG + BC regularizer)...")

        pbar = tqdm(
            total=self.config.get("n_iters"),
            desc=", ".join(
                [
                    f"episode_reward_mean_{group}=0"
                    for group in self.env.group_map.keys()
                ]
            ),
        )

        train_group_map = deepcopy(self.env.group_map)

        max_grad_norm = self.config.get("training").get("max_grad_norm")
        n_optimiser_steps = self.config.get("training").get("n_optimiser_steps")

        start_time = time.time()
        total_frames = 0
        total_episodes = 0
        total_train_steps = 0

        for iteration, batch in enumerate(self.collector):
            current_frames = batch.numel()
            total_frames += current_frames
            batch = self.process_batch(batch)

            bc_weight = self._bc_weight(iteration)

            actor_losses = {group: [] for group in self.env.group_map.keys()}
            critic_losses = {group: [] for group in self.env.group_map.keys()}
            bc_losses = {group: [] for group in self.env.group_map.keys()}

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

                for _ in range(n_optimiser_steps):
                    minibatch = self.replay_buffers[group].sample()

                    loss_vals = self.losses[group](minibatch)

                    actor_loss = loss_vals["loss_actor"]
                    value_loss = loss_vals["loss_value"]

                    actor_loss_val = actor_loss.item()
                    critic_loss_val = value_loss.item()

                    bc_loss_val = 0.0
                    if bc_weight > 0.0:
                        bc_loss = self._compute_bc_loss_for_group(group)
                        bc_loss_val = bc_loss.item()
                        actor_loss = actor_loss + bc_weight * bc_loss

                    actor_losses[group].append(actor_loss_val)
                    critic_losses[group].append(critic_loss_val)
                    bc_losses[group].append(bc_loss_val)

                    actor_optim = self.optimisers[group]["loss_actor"]
                    actor_loss.backward()
                    actor_params = actor_optim.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
                    actor_optim.step()
                    actor_optim.zero_grad()

                    value_optim = self.optimisers[group]["loss_value"]
                    value_loss.backward()
                    value_params = value_optim.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
                    value_optim.step()
                    value_optim.zero_grad()

                    self.target_updaters[group].step()
                    total_train_steps += 1

                self.exploration_policies[group][-1].step(current_frames)

            # --- Dedicated evaluation ---
            eval_means = self.evaluate(n_episodes=20)

            # --- Metrics ---
            elapsed = time.time() - start_time
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
                    eval_reward_mean=eval_means[group],
                    actor_loss=round(sum(actor_losses[group]) / n_opt, 6),
                    critic_loss=round(sum(critic_losses[group]) / n_opt, 6),
                    replay_size=len(self.replay_buffers[group]),
                    bc_loss=round(sum(bc_losses[group]) / n_opt, 6),
                    bc_weight=round(bc_weight, 6),
                )

            if iteration % 10 == 0:
                self.metrics_logger.save()

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

        first_group = list(self.env.group_map.keys())[0]
        recent = self.metrics_logger.get_values("episode_reward_mean", first_group)[-10:]
        return (
            f"RFT training complete. Environment: {self.config['scenario_name']}, "
            f"Experiment Type: {self.experiment_type}, Seed: {self.seed}.\n\n"
            f"Results: {recent}"
        )
