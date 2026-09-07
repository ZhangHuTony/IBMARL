'''
Replay buffer and data collection for IBMARL.

IBRL-style strategy: a single replay buffer per group with demos pre-loaded.
Online transitions are added during training and naturally dilute/overwrite
the demo data as the buffer fills.  No separate demo buffer, no annealing.
'''
from typing import Dict, Tuple

from tensordict.nn import TensorDictSequential
from tensordict import TensorDictBase, TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer


from src.experiments.ibmarl.modules import OverWriteActionWithBestComb, build_exploration_policy_with_noise

from pathlib import Path
import torch
import numpy as np


def _build_exploration_policy(parent, group, rl_policies, cfg, noise_modules_dict):
    """
    Build exploration policy for a group: clean RL policy -> noise -> arbiter.
    The noise module respects ExplorationType (no noise in MODE, noise in RANDOM).
    
    Args:
        noise_modules_dict: Dictionary to store noise module references for annealing
    """
    # Build policy with noise wrapper (noise respects ExplorationType)
    exploration_policy_with_noise = build_exploration_policy_with_noise(
        rl_policies[group],
        rl_policies[group].spec,
        cfg,
        group
    )
    
    # Store reference to noise module for annealing (it's the second module in the sequential)
    noise_modules_dict[group] = exploration_policy_with_noise[1]
    
    # Add arbiter on top
    return TensorDictSequential(
        exploration_policy_with_noise,
        OverWriteActionWithBestComb(parent.action_arbiter, group)
    )

def build_data_collector(cfg, parent, rl_policies, env, device):

    frames_per_batch = cfg.get('frames_per_batch')
    total_frames = cfg.get('total_frames')
    
    # Dictionary to store noise module references for annealing
    noise_modules = {}
    
    # Build exploration policies: clean RL -> noise (respects ExplorationType) -> arbiter
    exploration_policies = TensorDictSequential(
        *[_build_exploration_policy(parent, group, rl_policies, cfg, noise_modules) for group in env.group_map.keys()]
    )

    collector = SyncDataCollector(
        env,
        exploration_policies,
        frames_per_batch=frames_per_batch,
        device=device,
        total_frames=total_frames
    )

    # The collector must run THIS policy object, not a device-cast copy: a copy
    # carries its own ActionArbiter (critics frozen at initialisation, warm-up
    # flag never seen) and its own noise modules (annealing steps the originals).
    # torchrl copies whenever a parameter or buffer is off-device -- see
    # build_exploration_policy_with_noise for the buffer that used to trigger it.
    if collector.policy is not exploration_policies:
        raise RuntimeError(
            "SyncDataCollector copied the exploration policy instead of sharing "
            "it (a parameter or buffer is not on the collector device); the "
            "acting arbiter would score with stale critics."
        )

    return exploration_policies, collector, noise_modules

def _load_demonstrations_into_buffer(
    demonstration_path: Path, group: str, replay_buffer: ReplayBuffer, env
) -> None:
    """
    Load demonstrations into the given replay buffer, matching the
    [Batch, Agents, *] structure of the live collector (same as MADDPG/RLFD).
    """
    if not demonstration_path or not demonstration_path.exists():
        raise FileNotFoundError(f"Demonstrations not found at {demonstration_path}")

    print(f"[IBMARL] Loading demonstrations for group '{group}' from {demonstration_path}")
    demo_data = torch.load(
        demonstration_path, map_location="cpu", weights_only=False
    )

    obs = torch.tensor(np.array(demo_data["obs"]), dtype=torch.float32)
    act = torch.tensor(np.array(demo_data["act"]), dtype=torch.float32)
    rew = torch.tensor(np.array(demo_data["rewards"]), dtype=torch.float32)
    next_obs = torch.tensor(np.array(demo_data["next_obs"]), dtype=torch.float32)
    dones = torch.tensor(np.array(demo_data["dones"]), dtype=torch.bool)

    if rew.ndim == 2:
        rew = rew.unsqueeze(-1)

    total_elements = obs.shape[0]
    n_agents = obs.shape[1]

    # Episode ends in the demos come from the recorder's time limit, not true
    # termination, so bootstrap through them: done=True, terminated=False.
    done_agent = dones.view(-1, 1, 1).expand(total_elements, n_agents, 1)
    terminated_agent = torch.zeros((total_elements, n_agents, 1), dtype=torch.bool)

    agents_data = TensorDict(
        {
            "observation": obs,
            "action": act,
            "episode_reward": rew,
        },
        batch_size=[total_elements, n_agents],
    )

    # Built fresh rather than cloned from agents_data: cloning silently carried
    # the CURRENT observation into the next slot, making every demo transition a
    # self-transition and poisoning the TD target.
    next_agents_data = TensorDict(
        {
            "observation": next_obs,
            "episode_reward": rew.clone(),
            "done": done_agent.clone(),
            "terminated": terminated_agent.clone(),
            "reward": rew.clone(),
        },
        batch_size=[total_elements, n_agents],
    )

    next_td = TensorDict(
        {
            group: next_agents_data,
            "done": dones.view(-1, 1).clone(),
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
        f"[IBMARL] Loaded {total_elements} demonstration transitions into {group} buffer."
    )


def build_single_replay_buffer(
    cfg, env, device
) -> Dict[str, ReplayBuffer]:
    """
    IBRL-style single replay buffer per group with demos pre-loaded.
    Online transitions are added during training; old data (including demos)
    is naturally overwritten once the buffer fills.
    """
    demonstration_path = Path(cfg["demonstrations_path"])
    buffer_size = cfg.get("ibmarl_replay_size", cfg["memory_size"])
    train_batch_size = cfg["training"]["train_batch_size"]

    replay_buffers: Dict[str, ReplayBuffer] = {}

    for group, _agents in env.group_map.items():
        rb = ReplayBuffer(
            storage=LazyMemmapStorage(buffer_size),
            sampler=RandomSampler(),
            batch_size=train_batch_size,
        )
        if device.type != "cpu":
            rb.append_transform(lambda td: td.to(device))
        replay_buffers[group] = rb

        _load_demonstrations_into_buffer(demonstration_path, group, rb, env)

    return replay_buffers


def _concat_minibatches(td1: TensorDictBase, td2: TensorDictBase) -> TensorDictBase:
    """Concatenate two TensorDict minibatches along batch dimension (same as RLFD)."""
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


def build_demo_and_online_buffers(
    cfg, env, device
) -> Tuple[Dict[str, ReplayBuffer], Dict[str, ReplayBuffer]]:
    """
    Build demo (pre-loaded) and online (empty) replay buffers per group.
    Same storage/sampler/batch_size as MADDPG/RLFD.
    """
    demonstration_path = Path(cfg["demonstrations_path"])
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

        _load_demonstrations_into_buffer(
            demonstration_path, group, demo_buffers[group], env
        )

    return demo_buffers, online_buffers

def process_batch(env, batch: TensorDictBase) -> TensorDictBase:
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    for group in env.group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch