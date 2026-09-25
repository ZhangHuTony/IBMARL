"""
TO run this scenario:
    $  python3 -m src.demonstrations.synthetic.collect_policy_demos --scenario_name football --policy_type heuristic
"""
import numpy as np
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy

class SimpleBehindBallHeuristic(BaseHeuristicPolicy):

    def __init__(self, continuous_actions=True, align_factor=0.5):
        super().__init__(continuous_action=continuous_actions)
        self.align_factor = align_factor

    def compute_action(self, observations, u_range=None):
        actions = []

        for obs in observations:
            # Parse single-agent observation
            obs_np = np.array(obs, dtype=np.float32)
            my_x, my_y = obs_np[0], obs_np[1]
            ball_dx, ball_dy = obs_np[4], obs_np[5]
            ball_x, ball_y = my_x + ball_dx, my_y + ball_dy

            action = np.array([0.0, 0.0], dtype=np.float32)
            if my_x + 0.05 > ball_x:
                action[0] = -1.0
                action[1] =  (ball_y - my_y) / (abs(ball_y - my_y) + 1e-6)
            if my_x <= ball_x:
                action[0] = 1.0
                action[1] = (ball_y - my_y) / (abs(ball_y - my_y) + 1e-6)

            if u_range is not None:
                action = np.clip(action, -u_range, u_range)

            actions.append(action)

        return actions
