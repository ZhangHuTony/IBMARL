import numpy as np
import torch
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy

class SimpleBuzzwireHeuristic(BaseHeuristicPolicy):
    """
    2 agents and 1 ball in a corridor.
    Idea:
        1. Keep moving y closer to goal while keeping x away from walls.
    """
    def __init__(self, target_x_abs=0.6, move_factor=0.5, continuous_action=True):
        """
        :param target_x_abs: The absolute x-offset each agent tries to maintain from the center (0).
        param move_factor: How strongly the agent moves each step toward the target.
        """
        super().__init__(continuous_action=continuous_action)
        self.target_x_abs = target_x_abs
        self.move_factor = move_factor

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_size = observation.shape[0]
        device = observation.device
        agent_x = observation[:, 0]
        agent_y = observation[:, 1]
        rel_x = observation[:, 4]
        rel_y = observation[:, 5]

        # This keeps agents near left or right side, respectively, but not too close to corridor walls.
        # sign returns +1 for positive value, 0 for zero and -1 for negative value. We need this because by default, one agent spawns on the left(x<0) and other on right(x>0).
        # Adding 1e-9 as it prevents agent x from returning 0 as it returns weird agent behavior. :/
        sign = torch.sign(agent_x + 1e-9)
        desired_x = sign * self.target_x_abs
        goal_y = agent_y - rel_y

        action_x = self.move_factor * (desired_x - agent_x)
        action_y = self.move_factor * (goal_y - agent_y)
        actions = torch.stack([action_x, action_y], dim=-1)
        return actions

