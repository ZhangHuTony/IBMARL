# Balance Scenario Customization
_Last Updated: Oct 29, 2024_

The use of this package is to extend the capabilities of the VMAS simulator to allow for custom reward wrapping, scenario modification, dynamics modification, etc.

## Custom Reward Functions
Wrapping an environment with a custom reward function (i.e. something other than the ground truth reward) is commonplace in many IRL research projects.
VMAS does not offer explicit wrappers for this, but it is nonetheless straightforward to augment VMAS to allow for Observation-parameterized reward functions by inheriting from the [Balance Scenario class](https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/balance.py).

### Example: Injecting a Constant Reward Signal
As an example, the [ConstRewardBalanceScenario](wrappers/const_reward.py) purposefully replaces all GT rewards with 0.0 before they are sent to the RL algorithm.
The file simply inherits from the original Scenario and overrides the reward and info functions.

```python3
class ConstRewardBalanceScenario(BalanceScenario):
    def __init__(self):
        """
        Create the Scenario
        :param kwargs: Assumes the same set of arguments as vmas.scenarios.balance.BaseScenario.__init__
        """
        super().__init__()
        self.gt_reward = None

    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        """
        Given an agent in the scenario, return the reward of the agent.
        :param agent: the agent receiving the reward signal
        :return: the reward signal, a tensor of size num_envs
        """
        gt_reward = super().reward(agent)
        self.gt_reward = gt_reward  # Save the GT reward for evaluation

        # Modify Reward Here
        const_reward = gt_reward * 0.0
        return const_reward

    def info(self, agent: Agent):
        """
        Given an agent in the scenario, return the meta info of the agent.
        :param agent: the agent to extract info for
        :return: the info for the agent
        """
        info = super().info(agent)  # Grab the info from the superclass
        info["gt_reward"] = self.gt_reward
        return info
```

This code is a simple example that follows two principles. First, it does not disturb the dynamics or computation of the original environment whatsoever. It only modifies the reward return. Second, the notion of the Ground Truth Reward (GT_Reward) is still something that we want to keep track of during training, so it is saved step-wise to this class instance and returned as part of the info dictionary, which is then logged to WandB.

### Config
To indicate at training time that you want to train with a reward wrapper, you must add the `"wrapped_reward"` key to the environment config
```python3
"env_config": {
    ...
    "wrapped_reward": BalanceReward.CONSTANT,  # Use this to define custom reward functions
}
```

Here, the value of `"wrapped_reward"` is an enum of type [BalanceReward](../reward_enum.py). Simply adding this line to the config will evaluate RL under the constant 0.0 reward shown in the previous section.

### Extension
To add a new reward wrapper, use the following steps.
1. In [the wrappers directory](wrappers), create a new file.
2. Copy the contents of [const_reward.py](wrappers/const_reward.py) into the new file.
3. Modify the reward and info functions to your liking.
4. In [util.py](../reward_enum.py), add a key, value pair to the enum class [BalanceReward](../reward_enum.py), which will be used to instantiate your wrapper scenario.
5. In [wrapper_factory.py](../wrapper_factory.py), modify the condition tree to instantiate your new class if the input to the factory is the value of your new enum. See how this was done for BalanceReward.CONSTANT as an example.


### Testing
Run the training file, check WandB to see that the episode_reward_mean and the custom_metrics/agent_0/gt_reward_mean are showing different values.
See [this comment](https://github.com/Atharv-B/MARL-VMAS/issues/2#issuecomment-2444787295) for an example.

### Development Notes
During the development of this functionality, Github Issues was used to track the thought process and design decisions. See [Issue #2](https://github.com/Atharv-B/MARL-VMAS/issues/2) for more.