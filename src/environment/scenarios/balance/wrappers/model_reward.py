from vmas.scenarios.balance import Scenario as BalanceScenario
from vmas.simulator.core import Agent
from vmas.simulator.utils import AGENT_REWARD_TYPE
from src.rewards.reward_model import RewardModel
from src.scenarios.balance.util.observation_translator import ObsConcatTranslatorBalance
import numpy as np
import torch

class ModelRewardBalanceScenario(BalanceScenario):
    """
    class ModelRewardBalanceScenario wraps the VMAS Scenario
        with a learned reward function that inputs observations into the model and receives a scalar in return.
    The original GT reward is stored in a class variable that can be
        accessed at evaluation time.
    """
    def __init__(self, model: RewardModel):
        """
        Create the Scenario
        :param kwargs: Assumes the same set of arguments as vmas.scenarios.balance.BaseScenario.__init__
        """
        super().__init__()
        self.gt_reward = None
        self.model = model
        self.total_steps = 0
        self.most_recent_reward = torch.tensor([[0.0]])

    def reward(self, agent: Agent) -> AGENT_REWARD_TYPE:
        """
        Given an agent in the scenario, return the reward of the agent.
        :param agent: the agent receiving the reward signal
        :return: the reward signal, a tensor of size num_envs
        """
        is_first = agent == self.world.agents[0]
        gt_reward = super().reward(agent)

        if is_first:
            obs = torch.tensor([self.observation(agent).tolist() for agent in self.world.agents])
            obs = torch.swapaxes(obs, 0, 1).tolist()
            translated_obs = [ObsConcatTranslatorBalance.to_(raw_obs=np.array(o)).tolist() for o in obs]
            rew = self.model.r_hat(torch.tensor(translated_obs))
            self.most_recent_reward = rew.squeeze() if rew.numel() > 1 else rew

        self.gt_reward = gt_reward
        return self.most_recent_reward

    def info(self, agent: Agent):
        """
        Given an agent in the scenario, return the meta info of the agent.
        :param agent: the agent to extract info for
        :return: the info for the agent
        """
        info = super().info(agent)  # Grab the info from the superclass
        info["gt_reward"] = self.gt_reward
        return info

    def pre_step(self):
        self.total_steps += 1

    def done(self):
        """
        Set to False to prevent the simulation from ending when the ball hits the ground.
        """
        done = super().done()
        done.fill_(False)
        return done
