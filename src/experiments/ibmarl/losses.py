'''
handles losses and optimization
'''
import torch
import torch.nn.functional as F

import numpy as np

from tensordict import TensorDict, TensorDictBase



class GroupTrainer:
    def __init__(self, cfg, policies, critics, target_policies, target_critics, action_arbiter, env):

        #parameters
        self.lr = float(cfg["training"]["lr"])
        self.tau = float(cfg["training"]["polyak_tau"])
        self.max_grad_norm = float(cfg["training"]["max_grad_norm"])
        self.gamma =  float(cfg["training"]["gamma"])


        self.env = env
        self.action_arbiter = action_arbiter


        #networks
        self.rl_policies = policies
        self.critics = critics
        self.target_rl_policies = target_policies
        self.target_critics = target_critics

        self.num_critics = cfg["num_critics"]


        self.optimisers = self._build_optimisers()




    def update_critic(self, group, minibatch) -> dict: 
        '''
        returns losses and stats
        '''
        loss_value = self._ibmarl_value_loss(group, minibatch)


        # --- critic step ---
        opt_critic = self.optimisers[group]["loss_value"]
        opt_critic.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(opt_critic.param_groups[0]["params"], self.max_grad_norm)
        opt_critic.step()

        return None #TODO: have it return the losses
    
    def update_actor(self, group, minibatch) -> dict:
        loss_actor = self._ibmarl_actor_loss(group, minibatch)
        
        # --- actor step ---
        opt_actor = self.optimisers[group]["loss_actor"]
        opt_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(opt_actor.param_groups[0]["params"], self.max_grad_norm)
        opt_actor.step()


        return None #TODO: have it return the losses
    
    
    def polyak_step(self, source, target):
        '''
        updates target networks which handles the critic ensemble and the actor
        '''
        if isinstance(source, torch.nn.ModuleList):
            #iterate through the ensembles:
            for s_mod, t_mod in zip(source,target):
                self._single_polyak(s_mod, t_mod)
        else:
            self._single_polyak(source, target)


    def _single_polyak(self,source:torch.nn.Module, target: torch.nn.Module):
        '''
        updates target networks
        '''
        for p, p_targ in zip(source.parameters(), target.parameters()):
            p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
    
    def _build_optimisers(self):
        optimisers = {}
        for group in self.env.group_map.keys():
            critic_params = []
            for critic in self.critics[group]:
                critic_params.extend(list(critic.parameters()))
            
            optimisers[group]={
                "loss_actor": torch.optim.Adam(
                    self.rl_policies[group].parameters(),
                    lr = self.lr
                ),
                "loss_value" : torch.optim.Adam(
                    critic_params,
                    lr = self.lr
                )
            }
        return optimisers
    
    def _sample_indices(self):
        if self.num_critics <=2:
            return list(range(self.num_critics))
        return np.random.choice(self.num_critics, 2, replace = False)


    def _ibmarl_value_loss(self, group:str, mb: TensorDictBase) -> torch.Tensor:
        '''
        returns loss using bootstrap proposal, maintaining per-agent separation
        '''

        obs     = mb[(group, "observation")]
        act     = mb[(group, "action")]
        rew     = mb[("next", group, "reward")]  # Shape: [B, N, 1]
        done    = mb[("next", group, "done")]    # Shape: [B, N, 1] or [B, 1]
        next_obs= mb[("next", group, "observation")]

        # target calculation
        with torch.no_grad():
            # --------BOOTSTRAPPING PART------------------#
            # a_next_star: [B, N, Act_Dim]
            # q_next_val:  [B, N] (Minimized over 2 random critics, separate per agent)
            a_next_star, q_next_val = self.action_arbiter.bootstrap_proposal(group, next_obs)

           
            # Do NOT sum rewards. We want specific targets for specific agents.
            # Ensure shape is [B, N, 1]
            if rew.dim() < 3:
                 rew = rew.unsqueeze(-1)
            
            # Reshape q_next_val to match reward: [B, N] -> [B, N, 1]
            q_target_val = q_next_val.unsqueeze(-1)
            
            # Handle Done Mask
            # If done is shared ([B, 1]), broadcast it. 
            # If done is per-agent ([B, N, 1]), keep it.
            if done.dim() == 2: # [B, 1]
                 done = done.unsqueeze(-1) # [B, 1, 1]
            
            # Bellman Equation per Agent
            # Y shape: [B, N, 1]
            y = rew + self.gamma * (1.0 - done.float()) * q_target_val
        

        total_loss = 0
        td_cur = TensorDict({(group, "observation"): obs, (group, "action"): act}, batch_size=[obs.shape[0]], device = obs.device)

        for critic in self.critics[group]:
            # q shape: [B, N, 1]
            q = critic(td_cur)[(group, "state_action_value")]
            

            q_loss = F.mse_loss(q, y)
            
            total_loss += q_loss
        
        return total_loss
    
    def _ibmarl_actor_loss(self, group: str, mb: TensorDictBase) -> torch.Tensor:
        """
        MADDPG-style actor loss:
        L_actor = -E[ Q(obs, pi(obs)) ]

        mb[(group,"observation")] is [B, N, obs_dim]
        """
        obs = mb[(group, "observation")]  # [B,N,obs_dim]
        B = obs.shape[0]

        # Compute actions from current policy (NO exploration noise in the loss)
        td_pi = TensorDict({(group, "observation"): obs}, batch_size=[B], device=obs.device)
        td_pi = self.rl_policies[group](td_pi)
        a_pi = td_pi[(group, "action")]  # [B,N,act_dim]

        td_q = TensorDict({(group, "observation"): obs, (group, "action"): a_pi}, batch_size=[B], device=obs.device)

        q_list = []
        for i in range(self.num_critics):
            q = self.critics[group][i](td_q)[(group, "state_action_value")]
            q_list.append(q)
        
        q_stack = torch.stack(q_list, dim=0)
        q_min, _ = torch.min(q_stack, dim=0)

        # Reduce to a scalar per batch element
        if q_min.dim() == 3:
            q_tot = q_min.sum(dim=1).squeeze(-1)   # [B]
        elif q.dim() == 2:
            q_tot = q_min.squeeze(-1)              # [B]
        else:
            raise RuntimeError(f"Unexpected critic output shape in actor loss: {list(q.shape)}")

        return -q_tot.mean()
