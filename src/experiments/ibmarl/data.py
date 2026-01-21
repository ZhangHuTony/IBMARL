'''
holds the functions that manage the replay buffer and data collection
'''
from tensordict.nn import TensorDictSequential
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer


from src.experiments.ibmarl.modules import OverWriteActionWithBestComb


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
        replay_buffers[group] = replay_buffer
    return replay_buffers

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