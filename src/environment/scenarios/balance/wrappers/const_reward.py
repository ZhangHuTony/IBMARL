from vmas.scenarios.balance import Scenario as BalanceScenario
from vmas.simulator.core import Agent
from vmas.simulator.utils import AGENT_REWARD_TYPE

class ConstRewardBalanceScenario(BalanceScenario):
    """
    class BalanceConstantRewardWrappedScenario wraps the VMAS Scenario
        with a custom reward function that always returns the constant 0.
    The original GT reward is stored in a class variable that can be
        accessed at evaluation time.
    This allows us to test whether reward wrapping works in the context of VMAS
    """
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