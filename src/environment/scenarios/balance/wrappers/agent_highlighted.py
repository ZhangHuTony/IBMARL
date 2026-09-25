from triton.language import trans
from vmas.scenarios.balance import Scenario as BalanceScenario
from vmas.simulator.core import Agent
from vmas.simulator.utils import AGENT_REWARD_TYPE
# from src.rewards.reward_model import RewardModel
from src.scenarios.balance.util.observation_translator import ObsConcatTranslatorBalance
import numpy as np
from vmas.simulator.utils import Color
from vmas.simulator import rendering
import torch

class HighlightAgentBalanceScenario(BalanceScenario):
    """
    class HighlightAgentBalanceScenario wraps the VMAS Scenario
        with an additional render function that draws a circle over the agent being controlled.
    """
    def __init__(self, i=0):
        """
        Create the Scenario
        :param kwargs: Assumes the same set of arguments as vmas.scenarios.balance.BaseScenario.__init__
        """
        super().__init__()
        self.highlighted_agent = i

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        geoms = []
        for i, a in enumerate(self.world.agents):
            c = a.state.pos[env_index].tolist()
            circ = rendering.make_circle(0.03, res=30)
            xform = rendering.Transform(translation=c)
            circ.add_attr(xform)
            circ.set_color(*colors[i])
            geoms.append(circ)

        return geoms
