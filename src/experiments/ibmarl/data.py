'''
holds the functions that manage the replay buffer
'''

def build_exec_policy(env, exploration_policies, arbiter):
    ...

def build_data_collector(cfg, env, exec_policy, device):
    ...

def build_replay_buffer(cfg, env, device):
    ...

def process_batch(batch, env):
    ...