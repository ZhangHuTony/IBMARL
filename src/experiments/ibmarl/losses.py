'''
handles losses and optimization
'''

class GroupTrainer:
    def __init__(self):
        ...

    def update(self, group, minibatch) -> dict: 
        '''
        returns losses and stats
        '''
        ...

    def polyak_step(self,group):
        '''
        updates target networks
        '''
        ...