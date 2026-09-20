"""
Shared demonstration loader for the demo-based learners (IBMARL, RLfD, RFT).

Three byte-identical copies used to live in the three experiment modules; they
are folded here so the ``terminated`` handling has one home.
"""

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer


def load_demonstrations_into_buffer(
    demonstration_path: Path,
    group: str,
    replay_buffer: ReplayBuffer,
    *,
    log_prefix: str = "[demos]",
) -> None:
    """
    Load a recorded demonstration file into *replay_buffer*, matching the
    [Batch, Agents, *] layout of the live collector.

    ``dones`` marks episode ends.  ``terminated`` -- present in files written by
    analysis/relabel_demos_binary.py -- marks the subset of those that are true
    terminals, i.e. the step the task's success predicate fired, so the critic
    bootstraps through time-limit ends but not through successes.  Older
    recordings carry no ``terminated`` key: every episode end there comes from
    the recorder's time limit, so they load as all-False, exactly as before.
    """
    demonstration_path = Path(demonstration_path)
    if not demonstration_path.exists():
        raise FileNotFoundError(f"Demonstrations not found at {demonstration_path}")

    print(f"{log_prefix} Loading demonstrations for group '{group}' from {demonstration_path}")
    demo_data = torch.load(demonstration_path, map_location="cpu", weights_only=False)

    obs = torch.tensor(np.array(demo_data["obs"]), dtype=torch.float32)
    act = torch.tensor(np.array(demo_data["act"]), dtype=torch.float32)
    rew = torch.tensor(np.array(demo_data["rewards"]), dtype=torch.float32)
    next_obs = torch.tensor(np.array(demo_data["next_obs"]), dtype=torch.float32)
    dones = torch.tensor(np.array(demo_data["dones"]), dtype=torch.bool)
    if "terminated" in demo_data:
        terminated = torch.tensor(np.array(demo_data["terminated"]), dtype=torch.bool)
    else:
        terminated = torch.zeros_like(dones)

    if rew.ndim == 2:
        rew = rew.unsqueeze(-1)

    total_elements = obs.shape[0]
    n_agents = obs.shape[1]

    done_agent = dones.view(-1, 1, 1).expand(total_elements, n_agents, 1)
    terminated_agent = terminated.view(-1, 1, 1).expand(total_elements, n_agents, 1)

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
            "terminated": terminated.view(-1, 1).clone(),
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
        f"{log_prefix} Loaded {total_elements} demonstration transitions into {group} buffer "
        f"({int(terminated.sum())} terminal)."
    )
