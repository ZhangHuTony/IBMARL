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
         evaluates specific members of the target ensemble
         '''
         q_list = []
         for i in indices:
              q = self.target_critics[group][i](td)[(group, "state_action_value")]
              q_list.append(q)
              
         q_stack = torch.stack(q_list, dim = 0)

         return q_stack

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

        # Returns stacked Q-values: [2, K, B, N, 1]
        q_subset = self._evaluate_critics(group, td, indices)

        # Result: [K, B, N, 1]
        q_min_per_agent, _ = torch.min(q_subset, dim=0)

        q_tot = q_min_per_agent.sum(dim=2).squeeze(-1)
        
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

        # Compute metrics for logging
        # 1. Mean difference between a_RL and a_IL actions
        action_diff = (a_rl - a_il).abs()  # [B, N, act_dim]
        mean_action_diff = action_diff.mean().item()

        # 2. Mean and variance of q-value difference between a_RL and a_IL
        # Evaluate q-values for pure RL and pure IL actions
        # Find indices for all-RL (all 1s) and all-IL (all 0s) permutations
        all_rl_idx = choices.index(tuple([1] * N))
        all_il_idx = choices.index(tuple([0] * N))
        
        # Get q-values for all-RL and all-IL actions
        q_rl = q_min_per_agent[all_rl_idx].sum(dim=2).squeeze(-1)  # [B]
        q_il = q_min_per_agent[all_il_idx].sum(dim=2).squeeze(-1)  # [B]
        
        q_diff = q_rl - q_il  # [B]
        mean_q_diff = q_diff.mean().item()
        var_q_diff = q_diff.var().item()

        metrics = {
            'mean_action_diff': mean_action_diff,
            'mean_q_diff': mean_q_diff,
            'var_q_diff': var_q_diff,
        }

        return a_exec, mask, metrics
    
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
        q_subset = self._evaluate_critics(group, td_q, indices=indices) #should have shape [2, K, B, N, 1]

        q_min_per_agent, _ = torch.min(q_subset, dim=0) 

        q_vals = q_min_per_agent.squeeze(-1)

        q_sum_for_selection = q_vals.sum(dim=2)

        if self.soft:
            # Permute to [B, K] for Categorical distribution
            logits = q_sum_for_selection.permute(1, 0) / self.temperature
            dist = torch.distributions.Categorical(logits=logits)
            best_k = dist.sample() # [B]
        else:
            best_k = torch.argmax(q_sum_for_selection, dim=0) # [B]


        # Gather the best action: [B, N, Act_Dim]
        a_next_star = joint[best_k, torch.arange(B, device=next_obs.device)]
        
        # Gather the per-agent Q-values: [B, N]
        # Crucial: We use q_vals (separated), NOT q_sum_for_selection
        q_star = q_vals[best_k, torch.arange(B, device=next_obs.device)]

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

        # 1. Sample 2 random critics
        indices = self._sample_critic_indices()

        # 2. Evaluate ONLY those 2 critics
        # Returns stacked Q-values: [2, Candidates=2, B, N, 1]
        q_subset = self._evaluate_critics(group, td, indices)

        # 3. Take Min over the Ensemble subset (dim=0)
        # We want the conservative estimate for every agent in both candidates
        # Result: [Candidates=2, B, N, 1]
        q_min_per_agent, _ = torch.min(q_subset, dim=0)

        # 4. Aggregate for Selection (Sum over agents to pick best joint strategy)
        # Sum dim 2 (Agents) -> [2, B, 1] -> Squeeze -> [2, B]
        q_tot = q_min_per_agent.sum(dim=2).squeeze(-1)

        # -----------------------------------------------------------
        # Selection Logic
        # -----------------------------------------------------------

        if self.soft:
            # Permute to [B, 2] for Categorical distribution
            logits = q_tot.permute(1, 0) / self.temperature

            # Create distribution and sample (0 or 1)
            dist = torch.distributions.Categorical(logits=logits)
            best_k = dist.sample() # [B]
        else: 
            best_k = torch.argmax(q_tot, dim=0)  # [B], values in {0,1}

        # Gather the selected action
        a_exec = joint[best_k, torch.arange(B, device=obs.device)]  # [B,N,act_dim]

        # Create mask: 0 if IL was chosen, 1 if RL was chosen
        mask = best_k.view(-1, 1).expand(-1, N).float()

        # Compute metrics for logging
        # 1. Mean difference between a_RL and a_IL actions
        action_diff = (a_rl - a_il).abs()  # [B, N, act_dim]
        mean_action_diff = action_diff.mean().item()

        # 2. Mean and variance of q-value difference between a_RL and a_IL
        # q_tot[0] is all-IL, q_tot[1] is all-RL
        q_il = q_tot[0]  # [B]
        q_rl = q_tot[1]  # [B]
        
        q_diff = q_rl - q_il  # [B]
        mean_q_diff = q_diff.mean().item()
        var_q_diff = q_diff.var().item()

        metrics = {
            'mean_action_diff': mean_action_diff,
            'mean_q_diff': mean_q_diff,
            'var_q_diff': var_q_diff,
        }

        return a_exec, mask, metrics
    
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

        # A. Sample 2 indices
        indices = self._sample_critic_indices()
        
        # B. Evaluate only those 2 critics
        # Expected Output Shape: [2, Candidates=2, B, N, 1]
        q_subset = self._evaluate_critics(group, td_q, indices=indices)

        # C. Take Min over Ensemble Dimension (dim=0)
        # Result Shape: [Candidates=2, B, N, 1]
        q_min_per_agent, _ = torch.min(q_subset, dim=0)
        
        # Squeeze: [2, B, N]
        q_vals = q_min_per_agent.squeeze(-1)

        # ------------------------------------------------------------------
        # SELECTION
        # ------------------------------------------------------------------

        # Sum over agents for selection score
        # Shape: [2, B]
        q_sum_for_selection = q_vals.sum(dim=2)
        
        if self.soft:
            logits = q_sum_for_selection.permute(1, 0) / self.temperature
            dist = torch.distributions.Categorical(logits=logits)
            best_k = dist.sample() # [B]
        else: 
            best_k = torch.argmax(q_sum_for_selection, dim=0)  # [B], values in {0,1}

        # ------------------------------------------------------------------
        # RETURN
        # ------------------------------------------------------------------

        # Gather best action
        a_next_star = joint[best_k, torch.arange(B, device=next_obs.device)]
        
        # Gather per-agent Q-values (separated)
        q_star = q_vals[best_k, torch.arange(B, device=next_obs.device)]

        return a_next_star, q_star


    
