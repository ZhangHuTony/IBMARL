
from tensordict import TensorDictBase
import torch
import copy
from tqdm import tqdm

from src.experiments.base_marl_experiment import BaseMARLExperiment

from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
    AdditiveGaussianModule,
    AdditiveGaussianWrapper
)

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.objectives import DDPGLoss, ValueEstimators, SoftUpdate

from tensordict.nn import TensorDictModule, TensorDictSequential

class MaddpgExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)

        self.policies, self.exploration_policies = self._setup_policy()

        self.critics = self._setup_critic()

        self.agents_exploration_policy, self.collector = self._setup_data_collection()

        self.replay_buffers = self._setup_replay_buffer()

        self.losses, self.target_updaters, self.optimisers = self._setup_loss_functions()



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
            policy = ProbabilisticActor(
                module=policy_modules[group],
                spec=self.env.full_action_spec[group, "action"],
                in_keys = [(group, "param")],
                out_keys = [(group, "action")],
                distribution_class = TanhDelta,
                distribution_kwargs = {
                    "low": self.env.full_action_spec[group, "action"].space.low,
                    "high": self.env.full_action_spec[group, "action"].space.high,
                },
                return_log_prob = False,

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


    def train(self):
        print("Training MADDPG Experiment...")
        
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
                    loss_vals = self.losses[group](minibatch)

                    for loss_name in ["loss_actor", "loss_value"]:
                        loss = loss_vals[loss_name]
                        
                        optimiser = self.optimisers[group][loss_name]

                        loss.backward()

                        #Optional for some reason
                        params = optimiser.param_groups[0]['params']
                        torch.nn.utils.clip_grad_norm_(params, self.config.get('training').get('max_grad_norm'))

                        optimiser.step()
                        optimiser.zero_grad()

                    self.target_updaters[group].step()

                    # Annealing update for exploration noise
                self.exploration_policies[group].step(current_frames)
            
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



    def evaluate(self):
        print("Evaluating MADDPG Experiment...")
        # Implement MADDPG evaluation logic here
        pass

    def _setup_data_collection(self):
        # setup data collection logic here

        agents_exploration_policy = TensorDictSequential(*self.exploration_policies.values())

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
        
        target_updater = {
            group: SoftUpdate(loss, tau= self.config.get('training').get('polyak_tau')) for group, loss in losses.items()
        }

        optimisers = {
            group: {
                "loss_actor": torch.optim.Adam(
                    loss.actor_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
                ),
                "loss_critic": torch.optim.Adam(
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