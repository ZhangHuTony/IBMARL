from enum import Enum

class RewardType(Enum):
    """
    class RewardType wraps a python enum, functionally making this class just like the enums of C++ or Java
    You can reference the explict values with static accessors. e.g. >>> var = RewardType.NO_WRAP
    """
    NO_WRAP = 0         # Indicates no Reward Wrapping, the RL reward = GT reward
    CONSTANT = 1        # Indicates that the reward will always be a constant signal (used for testing)
    LEARNED_REWARD = 2  # Indicates that a learned model will be passed to the scenario to evaluate the reward

