
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


class IbmarlExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)



        #setup networks
        bc_path = Path(config["r2bc_checkpoint_path"])
        self.il_policy = R2bcPolicy(bc_path, self.env, self.device)

        self.policies, self.exploration_policies = build_rl_policies(config, self.env, self.device)

        self.critics = build_critics(config, self.env, self.device)

        self.target_policies, self.target_critics = build_targets(self.policies, self.critics, self.env)

        self.agents_exploration_policy, self.collector = self._setup_data_collection()


        self.replay_buffers = self._setup_replay_buffer()

        self.trainer = GroupTrainer(config, self.policies, self.critics, self.target_policies, self.target_critics, self.env)




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
            batch = self.process_batch(batch)

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

                    # print("reward", minibatch[("next","agents","reward")].shape)
                    # print("done", minibatch[("next","agents","done")].shape)
                    # print("obs", minibatch[("agents","observation")].shape)
                    # print("next_obs", minibatch[("next","agents","observation")].shape)
                    # print("act", minibatch[("agents","action")].shape)

                    self.trainer.update(group, minibatch) #TODO: save returns

                    self.trainer.polyak_step(self.policies[group], self.target_policies[group])
                    self.trainer.polyak_step(self.critics[group], self.target_critics[group])
                    
                    # loss_vals = self.losses[group](minibatch)
                    # for loss_name in ["loss_actor", "loss_value"]:
                        

                    #     loss = loss_vals[loss_name]
                        
                    #     optimiser = self.optimisers[group][loss_name]

                    #     loss.backward()

                    #     #Optional for some reason
                    #     params = optimiser.param_groups[0]['params']
                    #     torch.nn.utils.clip_grad_norm_(params, self.config.get('training').get('max_grad_norm'))

                    #     optimiser.step()
                    #     optimiser.zero_grad()

                    # self.target_updaters[group].step()

                    # Annealing update for exploration noise
                self.exploration_policies[group][-1].step(current_frames)


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
    
 



   
    


    def best_act_comb(self, group: str, obs: torch.Tensor, a_rl: torch.Tensor) -> torch.Tensor:
        '''
        finds the best joint action from {a_IL, a_RL}^N which maximizes critic value

        Input:
            obs:    [B, N, obs_dim]
            a_rl:   [B, N, act_dim] (RL action)
        
        Output:
            a_exec: [B, N, act_dim] 
        '''

        # dimension verifications
        if obs.dim() != 3:
            raise RuntimeError(f"obs must be [B,N,obs_dim], got {list(obs.shape)}")
        if a_rl.dim() != 3:
            raise RuntimeError(f"a_rl must be [B,N,act_dim], got {list(a_rl.shape)}")
        if obs.shape[:2] != a_rl.shape[:2]:
            raise RuntimeError(f"obs and a_rl batch/agent dims mismatch: obs {list(obs.shape)} vs a_rl {list(a_rl.shape)}")
        
        B, N, _ =       obs.shape
        _, _, act_dim = a_rl.shape

        #compute il action candidates
        a_il = self.il_policy.get_action(group, obs)


        if a_il.shape != a_rl.shape:
            raise RuntimeError(f"a_il shape {list(a_il.shape)} != a_rl shape {list(a_rl.shape)}")
        
        #build all permutations
        cand = torch.stack([a_il, a_rl], dim=0)

        choices = list(itertools.product([0,1], repeat=N))

        K = len(choices) #number of permutations

        joint = []

        for choice in choices:
            a_k = torch.stack([cand[choice[i], :, i, :] for i in range(N)], dim = 1)
            joint.append(a_k)
        
        joint = torch.stack(joint, dim = 0) # [K, B, N, act_dim]

        # score each permutation with critic
        td = TensorDict(
            {
                (group, "observation"): obs.unsqueeze(0).expand(K, B, *obs.shape[1:]),
                (group, "action") : joint
            },
            batch_size=[K, B],
            device = obs.device,
        )

        q = self.critics[group](td)[(group, "state_action_value")]

        #reduce q to value per permutation and environemt (k,b)
        # if value assigned per agent would be: [K,B,N,1]
        # if already aggregated [K,B, 1]
        if q.dim() == 4:
            q_tot = q.sum(dim=2).squeeze(-1)
        elif q.dim() == 3:
            q_tot = q.squeeze(-1)
        else:
            raise RuntimeError(f"unexpected critic output shape: {list(q.shape)}")
        
        best_k = torch.argmax(q_tot, dim=0)

        a_exec = joint[best_k, torch.arange(B, device=obs.device)]

        a_exec = a_rl
        return a_exec


    @torch.no_grad()
    def _test_best_perm_action(self):
        for group, agents in self.env.group_map.items():
            B = 4
            N = len(agents)
            obs_dim = self.env.observation_spec[group, "observation"].shape[-1]
            act_dim = self.env.full_action_spec[group, "action"].shape[-1]

            obs = torch.randn(B, N, obs_dim, device=self.device)
            a_rl = torch.randn(B, N, act_dim, device=self.device)

            a_exec = self.best_act_comb(group, obs, a_rl)
            assert a_exec.shape == (B, N, act_dim), f"got {a_exec.shape}"

            print(f"[best_perm_action OK] group={group}, a_exec shape={list(a_exec.shape)}")


    def make_group_exec_policy(self, group:str):
        return TensorDictSequential(
            self.exploration_policies[group],
            OverWriteActionWithBestComb(self,group),
        )

    def _setup_data_collection(self):
        # setup data collection logic here

        agents_exploration_policy = TensorDictSequential(
            *[self.make_group_exec_policy(group) for group in self.env.group_map.keys()]
        )

        collector = SyncDataCollector(
            self.env,
            agents_exploration_policy,
            frames_per_batch=self.config.get('frames_per_batch'),
            device=self.device,
            total_frames=self.config.get('total_frames'),
        )

        return agents_exploration_policy, collector

    def _setup_replay_buffer(self):
        # setup replay buffer logic here
        replay_buffers = {}
        for group, _agents in self.env.group_map.items():
            replay_buffer = ReplayBuffer(
                storage = LazyMemmapStorage(self.config.get('memory_size')), #must map to cpu
                sampler = RandomSampler(),
                batch_size = self.config.get('training').get('train_batch_size'),
            )

            if self.device.type != "cpu": #move to gpu if not training on cpu
                replay_buffer.append_transform(lambda td: td.to(self.device))
            replay_buffers[group] = replay_buffer
        return replay_buffers
    
    def _setup_loss_functions(self):
        # We no longer use DDPGLoss. We compute actor/value losses manually.
        optimisers = {
            group: {
                "loss_actor": torch.optim.Adam(
                    self.policies[group].parameters(),
                    lr=float(self.config["training"]["lr"]),
                ),
                "loss_value": torch.optim.Adam(
                    self.critics[group].parameters(),
                    lr=float(self.config["training"]["lr"]),
                ),
            }
            for group in self.env.group_map.keys()
        }

        # Keep return signature the same as your code expects.
        losses = None
        target_updater = None

        # losses = {}

        # for group, _agents in self.env.group_map.items():
        #     loss_module = DDPGLoss(
        #         actor_network = self.policies[group],
        #         value_network = self.critics[group],
        #         delay_value = True, #use target networks
        #     )
        #     loss_module.set_keys(
        #         state_action_value = (group, "state_action_value"),
        #         reward= (group, "reward"),
        #         done = (group, "done"),
        #         terminated = (group, "terminated"),
        #     )
        #     loss_module.make_value_estimator(ValueEstimators.TD0, gamma= self.config.get('training').get('gamma'))

        #     losses[group] = loss_module
        
        # target_updater = {
        #     group: SoftUpdate(loss, tau= self.config.get('training').get('polyak_tau')) for group, loss in losses.items()
        # }

        # optimisers = {
        #     group: {
        #         "loss_actor": torch.optim.Adam(
        #             loss.actor_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
        #         ),
        #         "loss_value": torch.optim.Adam(
        #             loss.value_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
        #         )
        #     }
        #     for group, loss in losses.items()
        # }

        return losses, target_updater, optimisers

    
    def process_batch(self, batch: TensorDictBase) -> TensorDictBase:
        """
        If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
        `"terminated"` and `"done"`.
        This is needed to present them with the same shape as the reward to the loss.
        """
        for group in self.env.group_map.keys():
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
    
    @torch.no_grad()
    def polyak_update_(self, source: torch.nn.Module, target: torch.nn.Module, tau: float):
        for p, p_targ in zip(source.parameters(), target.parameters()):
            p_targ.data.mul_(1.0 - tau).add_(tau * p.data)


    @torch.no_grad()
    def best_next_act_comb(self, group: str, next_obs: torch.Tensor) -> torch.Tensor:
        '''
        Used for bootstrap proposal
        '''

        B, N, obs_dim = next_obs.shape
        act_dim = self.env.full_action_spec[group, "action"].shape[-1]

        #IL candidate
        a_il = self.il_policy.get_action(group, next_obs)

        #target RL-candidate
        td_pi = TensorDict({(group, "observation"): next_obs}, batch_size=[B], device = next_obs.device)
        td_pi = self.target_policies[group](td_pi)
        a_rl = td_pi[(group, "action")]

        #build all combinations
        cand = torch.stack([a_il, a_rl], dim=0)
        choices = list(itertools.product([0,1], repeat=N))
        K = len(choices)

        joint=[]
        for choice in choices:
            a_k = torch.stack([cand[choice[i], :, i, :] for i in range(N)], dim = 1)
            joint.append(a_k)
        joint = torch.stack(joint, dim=0)

        #score each combination
        td_q = TensorDict(
            {
                (group, "observation"): next_obs.unsqueeze(0).expand(K, B, N, obs_dim),
                (group, "action"): joint,
            },
            batch_size=[K,B],
            device = next_obs.device,
        )

        q = self.target_critics[group](td_q)[(group, "state_action_value")]

        if q.dim() == 4:
            q_tot = q.sum(dim=2).squeeze(-1)
        elif q.dim() == 3:
            q_tot = q.squeeze(-1)
        else:
            raise RuntimeError(f"Unexpected target critic output shape: {list(q.shape)}")
        
        best_k = torch.argmax(q_tot, dim = 0)
        a_next_star = joint[best_k, torch.arange(B, device = next_obs.device)]
        return a_next_star
    

   