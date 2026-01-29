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
        obs = td[(g, "observation")] 
        a_rl = td[(g, "action")]


        a_exec, arbiter_choice_mask = self.parent.action_arbiter.actor_proposal(g, obs, a_rl)

        td[(g, "action")] = a_exec
        td[(g, "arbiter_choice")] = arbiter_choice_mask
        return td
    

