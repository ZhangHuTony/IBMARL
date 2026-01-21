
from src.experiments.base_marl_experiment import BaseMARLExperiment
from pathlib import Path

from src.r2bc.mabc import DecentralizedMiniBC


from tensordict import TensorDictBase
import torch
import copy
import itertools

import torch.nn.functional as F

from tqdm import tqdm


from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
    AdditiveGaussianModule,
)

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.objectives import DDPGLoss, ValueEstimators, SoftUpdate

from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict
from pathlib import Path


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder


from src.experiments.ibmarl.networks import R2bcPolicy, build_rl_policies, build_critics, build_targets
from src.experiments.ibmarl.modules import OverWriteActionWithBestComb
from src.experiments.ibmarl.losses import GroupTrainer
from src.experiments.ibmarl.arbiter import ActionArbiter
from src.experiments.ibmarl.data import build_data_collector, build_replay_buffer, process_batch


class IbmarlExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)



        #setup networks
        bc_path = Path(config["r2bc_checkpoint_path"])
        self.il_policies = R2bcPolicy(bc_path, self.env, self.device)

        self.rl_policies, self.rl_noise_policies = build_rl_policies(config, self.env, self.device)

        self.critics = build_critics(config, self.env, self.device)

        self.target_policies, self.target_critics = build_targets(self.rl_policies, self.critics, self.env)

        self.replay_buffers = build_replay_buffer(config, self.env, self.device)

        self.action_arbiter = ActionArbiter(self.il_policies, self.rl_policies, self.target_policies, self.critics, self.target_critics, self.env, self.device)

        self.agents_exploration_policy, self.collector = build_data_collector(config, self, self.rl_noise_policies, self.env, self.device)

        self.trainer = GroupTrainer(config, self.rl_policies, self.critics, self.target_policies, self.target_critics, self.action_arbiter, self.env)




    def train(self):
        print("Training IBMARL Experiment...")
        tau = float(self.config["training"]["polyak_tau"])

        pbar = tqdm(
            total= self.config.get('n_iters'),
            desc = ", ".join(
                [f"episode_reward_mean_{group}=0" for group in self.env.group_map.keys()]
            ), 
        )

        episode_reward_mean_map = {group: [] for group in self.env.group_map.keys()}
        train_group_map = copy.deepcopy(self.env.group_map)

        for iteration, batch in enumerate(self.collector):
            current_frames = batch.numel()
            batch = process_batch(self.env, batch)

            for group in train_group_map.keys():
                group_batch = batch.exclude(
                    *[
                        key 
                        for _group in self.env.group_map.keys()
                        if _group != group
                        for key in [_group, ("next", _group)]
                    ]
                ) #exclude other groups' data
                group_batch = group_batch.reshape(
                    -1
                ) 

                self.replay_buffers[group].extend(group_batch)

                for _ in range(self.config.get('training').get('n_optimiser_steps')):
                    minibatch = self.replay_buffers[group].sample()


                    self.trainer.update(group, minibatch) #TODO: save returns

                    self.trainer.polyak_step(self.rl_policies[group], self.target_policies[group])
                    self.trainer.polyak_step(self.critics[group], self.target_critics[group])
                    


                    # Annealing update for exploration noise
                self.rl_noise_policies[group][-1].step(current_frames)


            # Logging
            for group in self.env.group_map.keys():
                episode_reward_mean = (
                    batch.get(("next", group, "episode_reward"))[
                        batch.get(("next", group, "done"))
                    ]
                    .mean()
                    .item()
                    )
                
                episode_reward_mean_map[group].append(episode_reward_mean)

            pbar.set_description(
                ", ".join(
                    [
                        f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                        for group in self.env.group_map.keys()
                    ]
                ),
                refresh=False
            )
            pbar.update()

        self.results["group_map_keys"] = self.env.group_map.keys()
        self.results["episode_reward_mean_map"] = episode_reward_mean_map

    
    def save_checkpoint(self):
        raise NotImplementedError
    
    def render_policy(self):
        raise NotImplementedError
    
 


    




    

    




   