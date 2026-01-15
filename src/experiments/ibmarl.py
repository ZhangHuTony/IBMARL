
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


class OverWriteActionWithBestComb(torch.nn.Module):
    def __init__(self, parent, group:str):
        super().__init__()
        self.parent = parent
        self.group = group

    @torch.no_grad()
    def forward(self, td):
        g = self.group
        obs = td[(g,"observation")]
        a_rl = td[(g, "action")]

        a_exec = self.parent.best_act_comb(g, obs, a_rl)

        td[(g, "action")] = a_exec
        return td

class IbmarlExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)


        bc_path = Path(config["r2bc_checkpoint_path"])
        self.il_policy = self._initialize_Il_Policy(bc_path)
        self._smoke_test_il_policy()


        self.policies, self.exploration_policies = self._setup_policy()

        self.critics = self._setup_critic()

        self.target_policies, self.target_critics = self._setup_target_networks()

        self.agents_exploration_policy, self.collector = self._setup_data_collection()


        self.replay_buffers = self._setup_replay_buffer()

        self.losses, self.target_updaters, self.optimisers = self._setup_loss_functions()




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

                    loss_vals = self.losses[group](minibatch)


                    for loss_name in ["loss_actor", "loss_value"]:
                        
                        if loss_name == "loss_value":
                            loss = self.ibmarl_value_loss(group, minibatch)
                        else:
                            loss = loss_vals[loss_name]
                        
                        optimiser = self.optimisers[group][loss_name]

                        loss.backward()

                        #Optional for some reason
                        params = optimiser.param_groups[0]['params']
                        torch.nn.utils.clip_grad_norm_(params, self.config.get('training').get('max_grad_norm'))

                        optimiser.step()

                        optimiser.zero_grad()

                    # self.target_updaters[group].step() #should I keep this as an ablation?
                    self.polyak_update_(self.policies[group], self.target_policies[group], tau)
                    self.polyak_update_(self.critics[group],  self.target_critics[group],  tau)


                    # Annealing update for exploration noise
                self.exploration_policies[group][-1].step(current_frames)
            
            if iteration == self.config.get("horizon"):
                del train_group_map["agent"] #idk how deleting stops the training of that group but it does

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
    
    def _initialize_Il_Policy(self, r2bc_path):
        il = DecentralizedMiniBC.load_checkpoint(str(r2bc_path), device = self.device)
        il.eval()

        print("[IBMarl] Loaded R2BC IL policy.")
        print(f"[IBMarl] IL policy n={il.n}, in_size(total)={il.in_size*il.n}, out_size(total)={il.out_size*il.n}")
        return il
    
    @torch.no_grad()
    def _smoke_test_il_policy(self):
        '''
        test that dimension of r2bc policy lines up with environment
        '''
        for group, agents in self.env.group_map.items():
            B = 2
            N = len(agents)
            obs_dim = self.env.observation_spec[group, "observation"].shape[-1]
            act_dim = self.env.full_action_spec[group, "action"].shape[-1]

            obs = torch.randn(B, N, obs_dim, device=self.device)

            # concat to [B, N*obs_dim]
            x = obs.reshape(B, N * obs_dim)

            a_cat = self.il_policy(x)  # [B, N*?]
            if a_cat.shape[0] != B or a_cat.dim() != 2:
                raise RuntimeError(f"Unexpected IL output shape {a_cat.shape}")

            if a_cat.shape[1] != N * act_dim:
                raise RuntimeError(
                    f"IL action dim mismatch. IL outputs {a_cat.shape[1]} per batch, "
                    f"but env expects N*act_dim={N*act_dim} (N={N}, act_dim={act_dim})."
                )

            a = a_cat.reshape(B, N, act_dim)
            print(f"[IL OK] group={group}: obs {list(obs.shape)} -> act {list(a.shape)}")

    @torch.no_grad()
    def il_action(self, group:str, obs: torch.Tensor) -> torch.Tensor:
        B, N, O = obs.shape
        x = obs.reshape(B, N * O)
        a_cat = self.il_policy(x)
        act_dim = self.env.full_action_spec[group, 'action'].shape[-1]
        return a_cat.reshape(B, N, act_dim)



    def _setup_policy(self):
        policy_modules = {}
        print("Setting up MADDPG policy networks...")

        centralized = False #MADDPG uses decentralized policies (ADD THIS TO CONFIG LATER?)
        share_params = False #each agent has its own policy (ADD THIS TO CONFIG LATER?)

        # define neural network
        for group, agents in self.env.group_map.items():
            policy_net = MultiAgentMLP(
                n_agent_inputs= self.env.observation_spec[group, "observation"].shape[-1],
                n_agent_outputs= self.env.full_action_spec[group, "action"].shape[-1],
                n_agents = len(agents),
                centralized=centralized,
                share_params= share_params,
                device = self.device,
                depth = 2,
                num_cells = 256,
                activation_class= torch.nn.Tanh
            )

            policy_module = TensorDictModule(
                policy_net,
                in_keys=[(group, "observation")],
                out_keys=[(group, "param")],
            )

            policy_modules[group] = policy_module
        
        #wrap in probability distribution
        policies = {}

        for group, _agents in self.env.group_map.items():
            low = self.env.full_action_spec_unbatched[group, "action"].space.low.to(self.device)
            high = self.env.full_action_spec_unbatched[group, "action"].space.high.to(self.device)

            policy = ProbabilisticActor(
                module=policy_modules[group],
                spec=self.env.full_action_spec[group, "action"],
                in_keys=[(group, "param")],
                out_keys=[(group, "action")],
                distribution_class=TanhDelta,
                distribution_kwargs={"low": low, "high": high},
                return_log_prob=False,
            )

            policies[group] = policy
        
        #exploration policies
        exploration_policies = {}
        for group, _agents in self.env.group_map.items():
            print(self.config.get('total_frames') // 2) # type: ignore
            exploration_policy = TensorDictSequential(
                policies[group],
                AdditiveGaussianModule(
                    spec = policies[group].spec,
                    annealing_num_steps= self.config.get('total_frames') // 2, # type: ignore
                    action_key= (group, "action"),
                    sigma_init = 0.9,
                    sigma_end = 0.1,
                )
                
            )
            exploration_policies[group] = exploration_policy
        
        return policies, exploration_policies
    
    def _setup_critic(self):
        critics = {}

        share_critic_params = False #each agent has its own critic (ADD THIS TO CONFIG LATER?)
        centralized = True #MADDPG get priveleged information aka centralized critic (ADD THIS TO CONFIG LATER?)

        print("Setting up MADDPG critic networks...")
        for group, agents in self.env.group_map.items():
            
            cat_module = TensorDictModule(
                lambda obs, action: torch.cat([obs, action], dim=-1),
                in_keys=[(group, "observation"), (group, "action")],
                out_keys=[(group, "obs_action")],
            )

            critic_module = TensorDictModule(
                module = MultiAgentMLP(
                    n_agent_inputs= self.env.observation_spec[group, "observation"].shape[-1]
                    + self.env.full_action_spec[group, "action"].shape[-1],
                    n_agent_outputs=1,
                    n_agents = len(agents),
                    centralized=centralized,
                    share_params= share_critic_params,
                    device = self.device,
                    depth = 2,
                    num_cells = 256,
                    activation_class= torch.nn.Tanh
                ),
                in_keys=[(group, "obs_action")],
                out_keys=[(group, "state_action_value")],
            )

            critics[group] = TensorDictSequential(
                cat_module,
                critic_module
            )
        return critics

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
        a_il = self.il_action(group, obs)

        ##### this is a bandaid solution
        a_il = torch.clamp(a_il, -1.0, 1.0)
        #####
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
        # setup loss functions here
        losses = {}

        for group, _agents in self.env.group_map.items():
            loss_module = DDPGLoss(
                actor_network = self.policies[group],
                value_network = self.critics[group],
                delay_value = True, #use target networks
            )
            loss_module.set_keys(
                state_action_value = (group, "state_action_value"),
                reward= (group, "reward"),
                done = (group, "done"),
                terminated = (group, "terminated"),
            )
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma= self.config.get('training').get('gamma'))

            losses[group] = loss_module
        
        #target_updater = {
        #    group: SoftUpdate(loss, tau= self.config.get('training').get('polyak_tau')) for group, loss in losses.items() #keep for ablations?
        #}

        target_updater = None

        optimisers = {
            group: {
                "loss_actor": torch.optim.Adam(
                    loss.actor_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
                ),
                "loss_value": torch.optim.Adam(
                    loss.value_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
                )
            }
            for group, loss in losses.items()
        }

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
    
    def _setup_target_networks(self):

        target_policies =  {g: copy.deepcopy(self.policies[g]).eval() for g in self.env.group_map.keys()}
        target_critics =   {g: copy.deepcopy(self.critics[g]).eval() for g in self.env.group_map.keys()}

        for g in self.env.group_map.keys():
            for p in target_policies[g].parameters():
                p.requires_grad_(False)
            for p in target_critics[g].parameters():
                p.requires_grad_(False)
        
        return target_policies, target_critics

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
        a_il = self.il_action(group, next_obs)
        a_il = torch.clamp(a_il, -1.0, 1.0) #still not sure if this is the right thing to do

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
    

    def ibmarl_value_loss(self, group:str, mb: TensorDictBase) -> torch.Tensor:
        '''
        returns loss using bootstrap proposal
        '''

        obs     = mb[(group, "observation")]
        act     = mb[(group, "action")]
        rew     = mb[("next", group, "reward")]
        done    = mb[("next", group, "done")]
        next_obs= mb[("next", group, "observation")]

        gamma = float(self.config["training"]["gamma"])

        #current Q
        td_cur = TensorDict({(group, "observation"): obs, (group, "action"): act}, batch_size=[obs.shape[0]], device=obs.device)
        q = self.critics[group](td_cur)[(group, "state_action_value")]

        #target calculation
        with torch.no_grad():
            a_next_star = self.best_next_act_comb(group, next_obs)
            td_n = TensorDict({(group, "observation"): next_obs, (group, "action"): a_next_star},
                                batch_size=[next_obs.shape[0]], device=next_obs.device)
            q_next = self.target_critics[group](td_n)[(group, "state_action_value")]

            y = rew + gamma * (1.0-done.float()) * q_next
        

        #ensure dimensions align:
        if q.dim() == 2 and y.dim() == 3:
            y_red = y.sum(dim=1)
            loss = F.mse_loss(q, y_red)
        else:
            loss = F.mse_loss(q,y)
        
        return loss