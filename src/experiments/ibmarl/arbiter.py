'''
class used to find the best action combination given an IL policy and Critic
'''
import torch

class ActionArbiter:
    def __init__(self, il_policy, rl_policy, critics, env, device):
        pass


    def best_act_comb(self, group, obs, a_rl) -> torch.Tensor:
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


    def best_next_act_comb(self, group, next_obs, next_a_rl=None):
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
    