'''
class used to find the best action combination given an IL policy and Critic
'''
import torch
import itertools
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

        self.il_policy = il_policy
        self.rl_policy = rl_policy
        self.target_rl_policies = target_rl_policy
        self.critics = critics
        self.target_critics = target_critics
        self.env = env
        self.device = device

        self.metrics = {g: {"rl_action_count": 0, "total_action_count": 0} for g in env.group_map.keys()}
        

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
        

    def get_and_reset_metrics(self):
        """Calculates fraction of RL actions taken since last reset."""
        results = {}
        for group, data in self.metrics.items():
            if data["total_action_count"] > 0:
                frac = data["rl_action_count"] / data["total_action_count"]
            else:
                frac = 0.0
            results[group] = frac
            
            # Reset counters
            data["rl_action_count"] = 0
            data["total_action_count"] = 0
        return results


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

        q = self.target_critics[group](td)[(group, "state_action_value")]

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

        return a_rl, None

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

        q = self.target_critics[group](td)[(group, "state_action_value")]

        # Reduce to [2, B] so we can argmax over the 2 options
        if q.dim() == 4:          # [2,B,N,1]
            q_tot = q.sum(dim=2).squeeze(-1)  # [2,B]
        elif q.dim() == 3:        # [2,B,1]
            q_tot = q.squeeze(-1)            # [2,B]
        else:
            raise RuntimeError(f"unexpected critic output shape: {list(q.shape)}")

        ##greedy###
        #best_k = torch.argmax(q_tot, dim=0)  # [B], values in {0,1}
        ###########

        ##soft#####
        temperature = 1.0
        # 2. Transpose q_tot from [2, B] to [B, 2] for Categorical
        logits = q_tot.permute(1, 0) / temperature

        # 3. Create distribution and sample
        # This effectively performs softmax(logits) and samples index 0 or 1
        dist = torch.distributions.Categorical(logits=logits)
        
        best_k = dist.sample() # [B]

        ##############

        if group in self.metrics:
            # best_k is 1 if RL was chosen, 0 if IL. Sum gives total RL choices.
            self.metrics[group]["rl_action_count"] += best_k.sum().item()
            self.metrics[group]["total_action_count"] += best_k.numel()

        a_exec = joint[best_k, torch.arange(B, device=obs.device)]  # [B,N,act_dim]
        return a_exec, best_k
    
    def _best_next_act_strict(self, group, next_obs):
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

        #return a_rl #no-bootstrapping

        joint = torch.stack([a_il, a_rl], dim =0)

    
        td_q = TensorDict(
            {
                (group, "observation"): next_obs.unsqueeze(0).expand(2, B, N, obs_dim),
                (group, "action"): joint,
            },
            batch_size=[2,B],
            device = next_obs.device,
        )

        q = self.target_critics[group](td_q)[(group, "state_action_value")]

        if q.dim() == 4:
            q_tot = q.sum(dim=2).squeeze(-1)
        elif q.dim() == 3:
            q_tot = q.squeeze(-1)
        else:
            raise RuntimeError(f"Unexpected target critic output shape: {list(q.shape)}")
        
        ####greedy####
        #best_k = torch.argmax(q_tot, dim = 0)
        #########
        
        ##soft#####
        temperature = 1.0
        # 2. Transpose q_tot from [2, B] to [B, 2] for Categorical
        logits = q_tot.permute(1, 0) / temperature

        # 3. Create distribution and sample
        # This effectively performs softmax(logits) and samples index 0 or 1
        dist = torch.distributions.Categorical(logits=logits)
        
        best_k = dist.sample() # [B]

        ##############
        
        a_next_star = joint[best_k, torch.arange(B, device = next_obs.device)]

        return a_next_star
    


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

        return a_rl #no-bootstrapping

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
    
