import torch
import numpy as np
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
class SimpleTransportHeuristicPolicy(BaseHeuristicPolicy):
    """
    Agents line up directly behind the package in the goal direction and physically collide.
    Once close enough to contact, they push the package to the goal.
    """
    def __init__(
        self,
        agent_radius=0.03,
        package_radius=0.075,
        contact_eps=0.02,
        approach_thresh=0.05,
        continuous_action=True
    ):
        """
        :param agent_radius: radius of the agent (from scenario).
        :param package_radius: bounding 'radius' for the package (0.15 side => 0.075).
        :param contact_eps: small overlap margin so agents definitely collide.
        :param approach_thresh: how close to behind_pos before we push.
        """
        super().__init__(continuous_action=continuous_action)
        self.agent_r = agent_radius
        self.package_r = package_radius
        self.contact_eps = contact_eps
        self.approach_thresh = approach_thresh

    def compute_action(self, obs: torch.Tensor, u_range: float) -> torch.Tensor:
        """
        Observation layout for 1 package is:
            [agent_x, agent_y, agent_vx, agent_vy,
             (p-g)x, (p-g)y, (p-a)x, (p-a)y,
             package_vx, package_vy,
             on_goal_flag]
        where 'on_goal_flag' is obs[:, 10].
        """
        batch_size = obs.shape[0]
        device = obs.device
        on_goal_flag = obs[:, 10]                # shape=(batch_size,)
        # package->goal = -(p-g)
        package_goal_vec = -obs[:, 4:6]          # shape=[batch_size,2]
        # agent->package = obs[:, 6:8]
        agent_package_vec = obs[:, 6:8]          # shape=[batch_size,2]

        # agent pos
        agent_pos = obs[:, 0:2]
        package_pos = agent_pos + agent_package_vec

        actions = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)

        for i in range(batch_size):
            if on_goal_flag[i]:
                # Package on goal => do nothing
                actions[i] = torch.tensor([0.0, 0.0], device=device)
                continue

            # Compute normalized direction from package to goal
            p_to_g = package_goal_vec[i]
            dist_p_to_g = torch.norm(p_to_g)
            if dist_p_to_g < 1e-6:
                # If it's basically at goal, do nothing
                actions[i] = torch.tensor([0.0, 0.0], device=device)
                continue

            dir_p_to_g = p_to_g / dist_p_to_g

            # behind_pos = package_pos - (package_radius + agent_radius + contact_eps)*dir_p_to_g
            behind_dist = (self.package_r + self.agent_r + self.contact_eps)
            behind_pos = package_pos[i] - behind_dist * dir_p_to_g

            # agent->behind_pos
            behind_disp = behind_pos - agent_pos[i]
            dist_to_behind = torch.norm(behind_disp)

            if dist_to_behind > self.approach_thresh:
                # Approach behind_pos
                actions[i] = behind_disp
            else:
                # If close => push the package
                push_vec = p_to_g     # main push
                press_vec = agent_package_vec[i]  # small to keep us in contact
                combined = push_vec + 0.1 * press_vec
                actions[i] = combined

        actions = torch.clamp(actions, min=-u_range, max=u_range)
        return actions
