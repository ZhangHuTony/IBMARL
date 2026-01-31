'''
class used to find the best action combination given an IL policy and Critic
'''
import torch
import itertools
import numpy as np
from tensordict import TensorDict

class ActionArbiter:
    def __init__(
            self,
            config, 
            il_policy, 
            rl_policy, target_rl_policy, 
            critics, target_critics, 
            env, device):
        
        self.strict = config.get("strict")
        self.soft = config.get("soft")
        self.temperature = config.get("temperature")

        self.il_policy = il_policy
        self.rl_policy = rl_policy
        self.target_rl_policies = target_rl_policy
        self.critics = critics
        self.target_critics = target_critics
        self.env = env
        self.device = device

        self.num_critics = config.get("num_critics")

        

    def actor_proposal(self, group, obs, a_rl) -> torch.Tensor:
        if self.strict:
            return self._best_act_strict(group, obs, a_rl)
        else:
            return self._best_act_comb(group, obs, a_rl)
        
    def bootstrap_proposal(self, group, next_obs) -> torch.Tensor:
        if self.strict:
            return self._best_next_act_strict(group, next_obs)
        else:
            return self._best_next_act_comb(group, next_obs)
        

    def _sample_critic_indices(self):
         '''
         helper to sample 2 random critics from the ensemble
         '''
         if self.num_critics <=2:
              return list(range(self.num_critics))
         return np.random.choice(self.num_critics, 2, replace=False)
    
    def _evaluate_critics(self, group, td, indices):
         '''
         evaluates specific members of the target ensemble and returns the minimum Q-value
         '''
         q_list = []
         for i in indices:
              q = self.target_critics[group][i](td)[(group, "state_action_value")]
              q_list.append(q)
              
         q_stack = torch.stack(q_list, dim = 0)
         q_min, _ = torch.min(q_stack, dim=0)
         return q_min

    def _best_act_comb(self, group, obs, a_rl) -> torch.Tensor:
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
        
        #return a_rl #no IL Proposal

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

        indices = self._sample_critic_indices()
        q_min = self._evaluate_critics(group, td, indices)

        #reduce q to value per permutation and environemt (k,b)
        # if value assigned per agent would be: [K,B,N,1]
        # if already aggregated [K,B, 1]
        if q_min.dim() == 4:
            q_tot = q_min.sum(dim=2).squeeze(-1)
        elif q_min.dim() == 3:
            q_tot = q_min.squeeze(-1)
        else:
            raise RuntimeError(f"unexpected critic output shape: {list(q_min.shape)}")
        
        if self.soft:
                    # q_tot is [K, B]. Permute to [B, K] for Categorical distribution
                    logits = q_tot.permute(1, 0) / self.temperature
                    
                    # Create distribution over the K permutations and sample
                    dist = torch.distributions.Categorical(logits=logits)
                    best_k = dist.sample() # [B]
        else:
                    best_k = torch.argmax(q_tot, dim=0) # [B]

        a_exec = joint[best_k, torch.arange(B, device=obs.device)]

        perm_map = torch.tensor(choices, dtype=torch.float32, device=obs.device) # [K, N]
        mask = perm_map[best_k] # [B, N]


        return a_exec, mask
    
    def _best_next_act_comb(self, group, next_obs):
        '''
        Used for bootstrap proposal
        '''

        B, N, obs_dim = next_obs.shape
        act_dim = self.env.full_action_spec[group, "action"].shape[-1]

        #IL candidate
        a_il = self.il_policy.get_action(group, next_obs)


        #target RL-candidate
        td_pi = TensorDict({(group, "observation"): next_obs}, batch_size=[B], device = next_obs.device)
        td_pi = self.target_rl_policies[group](td_pi)
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

        indices = self._sample_critic_indices()
        q_min = self._evaluate_critics(group, td_q, indices=indices)

        if q_min.dim() == 4:
            q_tot = q_min.sum(dim=2).squeeze(-1)
        elif q_min.dim() == 3:
            q_tot = q_min.squeeze(-1)
        else:
            raise RuntimeError(f"Unexpected target critic output shape: {list(q_min.shape)}")
        
        if self.soft:
                    # q_tot is [K, B]. Permute to [B, K] for Categorical distribution
                    logits = q_tot.permute(1, 0) / self.temperature
                    
                    # Create distribution over the K permutations and sample
                    dist = torch.distributions.Categorical(logits=logits)
                    best_k = dist.sample() # [B]
        else:
                    best_k = torch.argmax(q_tot, dim=0) # [B]


        a_next_star = joint[best_k, torch.arange(B, device = next_obs.device)]
        q_star = q_tot[best_k, torch.arange(B, device=next_obs.device)]

        return a_next_star, q_star

    def _best_act_strict(self, group: str, obs: torch.Tensor, a_rl: torch.Tensor) -> torch.Tensor:
        """
        Choose between the all-IL joint action and the all-RL joint action
        by scoring both with the critic and taking the higher-value option.

        Inputs:
            obs:  [B, N, obs_dim]
            a_rl: [B, N, act_dim]

        Output:
            a_exec: [B, N, act_dim]
        """
        # dimension verifications
        if obs.dim() != 3:
            raise RuntimeError(f"obs must be [B,N,obs_dim], got {list(obs.shape)}")
        if a_rl.dim() != 3:
            raise RuntimeError(f"a_rl must be [B,N,act_dim], got {list(a_rl.shape)}")
        if obs.shape[:2] != a_rl.shape[:2]:
            raise RuntimeError(
                f"obs and a_rl batch/agent dims mismatch: obs {list(obs.shape)} vs a_rl {list(a_rl.shape)}"
            )

        B, N, _ = obs.shape

        # IL candidate (must match RL shape)
        a_il = self.il_policy.get_action(group, obs)
        if a_il.shape != a_rl.shape:
            raise RuntimeError(f"a_il shape {list(a_il.shape)} != a_rl shape {list(a_rl.shape)}")

        # Two candidates: k=0 -> all IL, k=1 -> all RL
        joint = torch.stack([a_il, a_rl], dim=0)  # [2, B, N, act_dim]

        # Score both candidates with critic in one forward pass
        td = TensorDict(
            {
                (group, "observation"): obs.unsqueeze(0).expand(2, B, *obs.shape[1:]),  # [2,B,N,obs_dim]
                (group, "action"): joint,                                             # [2,B,N,act_dim]
            },
            batch_size=[2, B],
            device=obs.device,
        )

        indices = self._sample_critic_indices()
        q_min = self._evaluate_critics(group, td, indices)

        # Reduce to [2, B] so we can argmax over the 2 options
        if q_min.dim() == 4:          # [2,B,N,1]
            q_tot = q_min.sum(dim=2).squeeze(-1)  # [2,B]
        elif q_min.dim() == 3:        # [2,B,1]
            q_tot = q_min.squeeze(-1)            # [2,B]
        else:
            raise RuntimeError(f"unexpected critic output shape: {list(q_min.shape)}")



        if self.soft:
      
            logits = q_tot.permute(1, 0) / self.temperature

            # 3. Create distribution and sample
            # This effectively performs softmax(logits) and samples index 0 or 1
            dist = torch.distributions.Categorical(logits=logits)
            
            best_k = dist.sample() # [B]

        else: 
            best_k = torch.argmax(q_tot, dim=0)  # [B], values in {0,1}

    

        a_exec = joint[best_k, torch.arange(B, device=obs.device)]  # [B,N,act_dim]

        mask = best_k.view(-1, 1).expand(-1, N).float()

        return a_exec, mask
    
    def _best_next_act_strict(self, group, next_obs):
        '''
        Used for bootstrap proposal
        '''

        B, N, obs_dim = next_obs.shape

        act_dim = self.env.full_action_spec[group, "action"]
        
        #IL candidate
        a_il = self.il_policy.get_action(group, next_obs)


        #target RL-candidate
        td_pi = TensorDict({(group, "observation"): next_obs}, batch_size=[B], device = next_obs.device)
        td_pi = self.target_rl_policies[group](td_pi)
        a_rl = td_pi[(group, "action")]

        joint = torch.stack([a_il, a_rl], dim =0)

    
        td_q = TensorDict(
            {
                (group, "observation"): next_obs.unsqueeze(0).expand(2, B, N, obs_dim),
                (group, "action"): joint,
            },
            batch_size=[2,B],
            device = next_obs.device,
        )

        indices = self._sample_critic_indices()
        q_min = self._evaluate_critics(group, td_q, indices=indices)

        if q_min.dim() == 4:
            q_tot = q_min.sum(dim=2).squeeze(-1)
        elif q_min.dim() == 3:
            q_tot = q_min.squeeze(-1)
        else:
            raise RuntimeError(f"Unexpected target critic output shape: {list(q_min.shape)}")
        
        if self.soft:
      
            logits = q_tot.permute(1, 0) / self.temperature

            # 3. Create distribution and sample
            # This effectively performs softmax(logits) and samples index 0 or 1
            dist = torch.distributions.Categorical(logits=logits)
            
            best_k = dist.sample() # [B]

        else: 
            best_k = torch.argmax(q_tot, dim=0)  # [B], values in {0,1}

        a_next_star = joint[best_k, torch.arange(B, device = next_obs.device)]
        q_star = q_tot[best_k, torch.arange(B, device=next_obs.device)]


        return a_next_star, q_star
    



    
