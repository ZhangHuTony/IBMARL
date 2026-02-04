'''
builds the Neural Nets needed for IBMARL
'''

from src.r2bc.mabc import DecentralizedMiniBC



import torch
import torch.nn as nn
import copy

from torchrl.modules import(
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
    AdditiveGaussianModule
)

from tensordict.nn import TensorDictModule, TensorDictSequential

class R2bcPolicy():
    def __init__(self, policy_path, env, device):

        self.device = device
        self.env = env

        self.policy = self._load_policy(policy_path)

        self._smoke_test_il_policy()

    def get_action(self, group:str, obs: torch.Tensor) -> torch.Tensor:
        B, N, O = obs.shape
        x = obs.reshape(B, N * O)
        a_cat = self.policy(x)
        act_dim = self.env.full_action_spec[group, 'action'].shape[-1]
        a = a_cat.reshape(B, N, act_dim)
        return torch.clamp(a, -1.0, 1.0)
    
    def _load_policy(self, policy_path):

        print("Loading IL Policy...")
        policy = DecentralizedMiniBC.load_checkpoint(str(policy_path), device = self.device)
        return policy.eval()
    
        
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


            a = self.get_action(group, obs)  # [B, N*?]
            if a.shape != (B, N, act_dim):
                raise RuntimeError(f"Unexpected IL output shape {a.shape}, expected {(B, N, act_dim)}")

            print(f"[IL OK] group={group}: obs {list(obs.shape)} -> act {list(a.shape)}")
    
    def get_average_reward(self, num_episodes: int = 30):
        '''
        Get the average return of just running the il policy on the environment
        '''
        raise NotImplementedError
    
    def render(self):
        '''
        create a rendering of a rollout of the policy
        '''
        raise NotImplementedError
   


def build_rl_policies(cfg, env, device):
    print("Setting up IBMARL RL Policies...")

    policy_modules= {}

    centralized = False #each agent's actor can only see their own observations
    share_params = False #each agent has their own policy

    for group, agents in env.group_map.items():
        policy_net = MultiAgentMLP(
            n_agent_inputs= env.observation_spec[group, "observation"].shape[-1],
            n_agent_outputs= env.full_action_spec[group, "action"].shape[-1],
            n_agents = len(agents),
            centralized=centralized,
            share_params= share_params,
            device = device,
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

    for group, _agents in env.group_map.items():
        low = env.full_action_spec_unbatched[group, "action"].space.low.to(device)
        high = env.full_action_spec_unbatched[group, "action"].space.high.to(device)

        policy = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.full_action_spec[group, "action"],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            distribution_class=TanhDelta,
            distribution_kwargs={"low": low, "high": high},
            return_log_prob=False,
        )

        policies[group] = policy
    
    #exploration policies
    exploration_policies = {}
    for group, _agents in env.group_map.items():
        exploration_policy = TensorDictSequential(
            policies[group],
            AdditiveGaussianModule(
                spec = policies[group].spec,
                annealing_num_steps= cfg.get('total_frames') // 2, # type: ignore
                action_key= (group, "action"),
                sigma_init = 0.1,
                sigma_end = 0.1,
            )
            
        )
        exploration_policies[group] = exploration_policy
    
    return policies, exploration_policies


def build_critics(cfg, env, device):
    print("Setting up IBMARL critic networks...")

    ensemble_size = cfg.get('num_critics')

    critics = {}
    
    share_critic_params = False #each agent has their own critic
    centralized = True # agent's critics can see other agent's observations

    for group, agents in env.group_map.items():
            
            group_ensemble = nn.ModuleList()

            for _ in range(ensemble_size):
            
                cat_module = TensorDictModule(
                    lambda obs, action: torch.cat([obs, action], dim=-1),
                    in_keys=[(group, "observation"), (group, "action")],
                    out_keys=[(group, "obs_action")],
                )

                critic_module = TensorDictModule(
                    module = MultiAgentMLP(
                        n_agent_inputs= env.observation_spec[group, "observation"].shape[-1]
                        + env.full_action_spec[group, "action"].shape[-1],
                        n_agent_outputs=1,
                        n_agents = len(agents),
                        centralized=centralized,
                        share_params= share_critic_params,
                        device = device,
                        depth = 2,
                        num_cells = 256,
                        activation_class= torch.nn.Tanh
                    ),
                    in_keys=[(group, "obs_action")],
                    out_keys=[(group, "state_action_value")],
                )

                

                ensemble_member = TensorDictSequential(
                    cat_module,
                    critic_module
                )
                group_ensemble.append(ensemble_member)
            
            critics[group] = group_ensemble
    
    return critics

    


def build_targets(policies, critics, env):
    
    target_policies = {g: copy.deepcopy(policies[g]).eval() for g in env.group_map.keys()}
    target_critics =   {g: copy.deepcopy(critics[g]).eval() for g in env.group_map.keys()}

    for g in env.group_map.keys():
        for p in target_policies[g].parameters():
            p.requires_grad_(False)

        for critic_member in target_critics[g]:
            for p in critic_member.parameters():
                p.requires_grad_(False)
    
    return target_policies, target_critics
