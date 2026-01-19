'''
holds custom torch module needed for IBMARL
'''

import torch

class OverWriteActionWithBestComb(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ...
    
    def forward(self, td):
        ...