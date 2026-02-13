'''
holds the functions that manage the replay buffer and data collection
'''
from tensordict.nn import TensorDictSequential
from tensordict import TensorDictBase, TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer


from src.experiments.ibmarl.modules import OverWriteActionWithBestComb

from pathlib import Path
import torch
import numpy as np


def _build_exploration_policy(parent, group, exploration_policies):
    return TensorDictSequential(
        exploration_policies[group],
        OverWriteActionWithBestComb(parent ,group)
    )

def build_data_collector(cfg, parent, rl_noise_policies, env, device):

    frames_per_batch = cfg.get('frames_per_batch')
    total_frames = cfg.get('total_frames')
    
    exploration_policies = TensorDictSequential(
        *[_build_exploration_policy(parent, group, rl_noise_policies) for group in env.group_map.keys()]
    )

    collector = SyncDataCollector(
        env,
        exploration_policies,
        frames_per_batch=frames_per_batch,
        device=device,
        total_frames=total_frames
    )

    return exploration_policies, collector

def build_replay_buffer(cfg, env, device):
    demonstration_path = cfg.get('demonstrations_path')
    memory_size = cfg.get('memory_size')
    batch_size = cfg.get('training').get('train_batch_size')

    # setup replay buffer logic here
    replay_buffers = {}
    for group, _agents in env.group_map.items():
        replay_buffer = ReplayBuffer(
            storage = LazyMemmapStorage(memory_size), #must map to cpu
            sampler = RandomSampler(),
            batch_size = batch_size,
        )

        
        if device.type != "cpu": #move to gpu if not training on cpu
            replay_buffer.append_transform(lambda td: td.to(device))

        _load_demonstrations(demonstration_path, group, replay_buffer)

        replay_buffers[group] = replay_buffer
    return replay_buffers

def _load_demonstrations(demonstration_path, group, replay_buffer):
    """
    Loads demonstrations into the buffer, matching the [Batch, Agents, 1] structure
    of the live collector.
    """
    if not demonstration_path or not Path(demonstration_path).exists():
        raise Exception("Demonstrations not found at {demonstration_path}")

    print(f"Loading demonstrations for group '{group}' from {demonstration_path}")
    try:
        # 1. Load data
        demo_data = torch.load(demonstration_path, map_location="cpu", weights_only=False)
        
        obs = torch.tensor(np.array(demo_data["obs"]), dtype=torch.float32)      # [T, 3, 18]
        act = torch.tensor(np.array(demo_data["act"]), dtype=torch.float32)      # [T, 3, 2]
        rew = torch.tensor(np.array(demo_data["rewards"]), dtype=torch.float32)  # [T, 3] or [T, 3, 1]
        
        # --- FIX: UNSQUEEZE Rewards to match [T, 3, 1] ---
        # The live collector uses [3, 1], so we must ensure the last dim is 1.
        if rew.ndim == 2:  # If shape is [T, 3]
            rew = rew.unsqueeze(-1)  # Becomes [T, 3, 1]
            
        total_elements = obs.shape[0]

        # 2. Build the 'agents' TensorDict 
        # Matches fields={agents: ...} in your printout
        agents_data = TensorDict({
            "observation": obs,
            "action": act,
            "episode_reward": rew, # Now guaranteed to be [T, 3, 1]
        }, batch_size=[total_elements, 3]) # Matches 'agents' batch_size=[3] from printout

        # 3. Build the 'next' TensorDict
        # Your printout shows 'next' contains 'agents' with its own done/reward
        next_agents_data = agents_data.clone()
        
        # Add agent-level done to 'next -> agents' (as seen in your printout)
        next_agents_data.set("done", torch.zeros((total_elements, 3, 1), dtype=torch.bool))
        next_agents_data.set("terminated", torch.zeros((total_elements, 3, 1), dtype=torch.bool))
        next_agents_data.set("reward", rew.clone()) # Often 'reward' exists alongside 'episode_reward'

        next_td = TensorDict({
            group: next_agents_data,
            "done": torch.zeros((total_elements, 1), dtype=torch.bool),      # Global done
            "terminated": torch.zeros((total_elements, 1), dtype=torch.bool), # Global terminated
        }, batch_size=[total_elements])

        # 4. Final Top-Level TensorDict matches the collector output
        td = TensorDict({
            group: agents_data,
            "next": next_td,
            "done": torch.zeros((total_elements, 1), dtype=torch.bool),
            "terminated": torch.zeros((total_elements, 1), dtype=torch.bool),
            "collector": TensorDict({
                 "traj_ids": torch.zeros((total_elements,), dtype=torch.int64)
            }, batch_size=[total_elements])
        }, batch_size=[total_elements])

        # 5. Extend the buffer
        replay_buffer.extend(td)
        print(f"Successfully loaded {total_elements} transitions into {group} buffer with shape {rew.shape}.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception("Failed to load demonstrations for group {group}: {e}")

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