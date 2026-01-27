'''
handles losses and optimization
'''
import torch
import torch.nn.functional as F

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


        self.optimisers = self._build_optimisers()




    def update(self, group, minibatch) -> dict: 
        '''
        returns losses and stats
        '''
        loss_actor = self._ibmarl_actor_loss(group, minibatch)
        loss_value = self._ibmarl_value_loss(group, minibatch)

        # --- actor step ---
        opt_actor = self.optimisers[group]["loss_actor"]
        opt_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(opt_actor.param_groups[0]["params"], self.max_grad_norm)
        opt_actor.step()

        # --- critic step ---
        opt_critic = self.optimisers[group]["loss_value"]
        opt_critic.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(opt_critic.param_groups[0]["params"], self.max_grad_norm)
        opt_critic.step()

        return None #TODO: have it return the losses
    

    def polyak_step(self,source:torch.nn.Module, target: torch.nn.Module):
        '''
        updates target networks
        '''
        for p, p_targ in zip(source.parameters(), target.parameters()):
            p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
    
    def _build_optimisers(self):
        optimisers = {
            group: {
                "loss_actor": torch.optim.Adam(
                    self.rl_policies[group].parameters(),
                    lr=self.lr,
                ),
                "loss_value": torch.optim.Adam(
                    self.critics[group].parameters(),
                    lr=self.lr
                ),
            }
            for group in self.env.group_map.keys()
        }

        return optimisers


    #TODO:
    def _ibmarl_value_loss(self, group:str, mb: TensorDictBase) -> torch.Tensor:
        '''
        returns loss using bootstrap proposal
        '''

        obs     = mb[(group, "observation")]
        act     = mb[(group, "action")]
        rew     = mb[("next", group, "reward")]
        done    = mb[("next", group, "done")]
        next_obs= mb[("next", group, "observation")]

        #current Q
        td_cur = TensorDict({(group, "observation"): obs, (group, "action"): act}, batch_size=[obs.shape[0]], device=obs.device)
        q = self.critics[group](td_cur)[(group, "state_action_value")]

        #target calculation
        with torch.no_grad():
            #--------BOOTSTRAPPING PART------------------#
            a_next_star = self.action_arbiter.bootstrap_proposal(group, next_obs)
            #-------------------------------------------------#
            td_n = TensorDict({(group, "observation"): next_obs, (group, "action"): a_next_star},
                                batch_size=[next_obs.shape[0]], device=next_obs.device)
            q_next = self.target_critics[group](td_n)[(group, "state_action_value")]

            y = rew + self.gamma * (1.0-done.float()) * q_next
        

        

        #ensure dimensions align:
        if q.dim() == 2 and y.dim() == 3:
            y_red = y.sum(dim=1)
            loss = F.mse_loss(q, y_red)
        else:
            loss = F.mse_loss(q,y)
        
        return loss
    
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

        # Evaluate critic on (obs, a_pi)
        td_q = TensorDict({(group, "observation"): obs, (group, "action"): a_pi}, batch_size=[B], device=obs.device)
        q = self.critics[group](td_q)[(group, "state_action_value")]  # [B,N,1] or [B,1]

        # Reduce to a scalar per batch element
        if q.dim() == 3:
            q_tot = q.sum(dim=1).squeeze(-1)   # [B]
        elif q.dim() == 2:
            q_tot = q.squeeze(-1)              # [B]
        else:
            raise RuntimeError(f"Unexpected critic output shape in actor loss: {list(q.shape)}")

        return -q_tot.mean()
