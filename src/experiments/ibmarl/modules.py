'''
holds custom torch module needed for IBMARL
'''

import torch

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

        a_exec = self.parent.action_arbiter.best_act_comb(g, obs, a_rl)
        #a_exec = self.parent.action_arbiter.best_act_strict(g, obs, a_rl)

        td[(g, "action")] = a_exec
        return td