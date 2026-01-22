'''
holds custom torch module needed for IBMARL
'''

import torch
import torch.nn as nn

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

        a_exec, arbiter_choice = self.parent.action_arbiter.actor_proposal(g, obs, a_rl)

        td[(g, "action")] = a_exec

        B, N, _ = obs.shape
        arbiter_choice_expanded = arbiter_choice.unsqueeze(-1).expand(B, N)

        td[(g, "arbiter_choice")] = arbiter_choice_expanded
        return td
    

class IndependentAgentPolicy(nn.Module):
    """
    Creates a separate MLP for each agent (share_params=False), 
    following the exact architecture from the IBRL paper (Fig 19):
    Linear -> LayerNorm -> Dropout -> ReLU
    """
    def __init__(self, n_agents, input_dim, output_dim, hidden_dim=256, depth=3, dropout=0.5):
        super().__init__()
        self.agents = nn.ModuleList([
            self._build_mlp(input_dim, output_dim, hidden_dim, depth, dropout)
            for _ in range(n_agents)
        ])

    def _build_mlp(self, in_dim, out_dim, hidden_dim, depth, dropout):
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        
        # Hidden Layers (depth - 1 intermediate layers)
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            
        # Output Layer (Pure Linear, Tanh handled by ProbabilisticActor later)
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        return nn.Sequential(*layers)

    def forward(self, obs):
        # obs shape: [Batch, n_agents, obs_dim]
        # We process each agent's observation with its own network
        outputs = []
        for i, net in enumerate(self.agents):
            # Slice the observation for agent i: [Batch, obs_dim]
            agent_obs = obs[:, i, :]
            out = net(agent_obs)
            outputs.append(out)
        
        # Stack back to [Batch, n_agents, act_dim]
        return torch.stack(outputs, dim=1)