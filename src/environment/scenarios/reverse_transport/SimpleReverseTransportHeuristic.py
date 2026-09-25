import torch
import numpy as np
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy

class SimpleReverseTransportHeuristic(BaseHeuristicPolicy):
    """
    Agents inside a large hollow box.
      1) Move to an 'inner edge' in the direction of (package->goal).
      2) Once close, push the box outward to the goal while staying in contact.
    """

    def __init__(
        self,
        box_halfwidth=0.3,
        agent_radius=0.03,
        margin=0.01,
        approach_thresh=0.05,
        pressing_factor=0.1,
        continuous_action=True
    ):
        """
        :param box_halfwidth: half the box side (0.3 if length=0.6).
        :param agent_radius: agent radius, ~0.03 in scenario.
        :param margin: small clearance so agent remains inside the box.
        :param approach_thresh: distance to 'edge_pos' at which we switch from approach to push.
        :param pressing_factor: how strongly to add (agent->package) while pushing.
        """
        super().__init__(continuous_action=continuous_action)
        self.box_halfwidth = box_halfwidth
        self.agent_r = agent_radius
        self.margin = margin
        self.approach_thresh = approach_thresh
        self.pressing_factor = pressing_factor

    def compute_action(self, obs: torch.Tensor, u_range: float) -> torch.Tensor:
        """
        Observations for reverse_transport scenario:
          [ agent_x, agent_y, agent_vx, agent_vy,
            package_vx, package_vy,
            (package_x - agent_x), (package_y - agent_y),
            (package_x - goal_x), (package_y - goal_y) ]
        No direct 'on_goal_flag'

        Approach:
          1) Approach an inside edge.
          2) Push once close.
        """
        batch_size = obs.shape[0]
        device = obs.device
        agent_package_vec = obs[:, 6:8]
        package_goal_vec = -obs[:, 8:10]
        agent_pos = obs[:, 0:2]
        package_pos = agent_pos + agent_package_vec
        dist_p2g = torch.linalg.norm(package_goal_vec, dim=-1)

        actions = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)

        for i in range(batch_size):
            if dist_p2g[i] < 1e-6:
                actions[i] = torch.tensor([0.0, 0.0], device=device)
                continue
            dir_p2g = package_goal_vec[i] / dist_p2g[i]
            inner_radius = self.box_halfwidth - self.agent_r - self.margin
            edge_pos = package_pos[i] + inner_radius * dir_p2g
            agent_edge_vec = edge_pos - agent_pos[i]
            dist_edge = torch.norm(agent_edge_vec)

            if dist_edge > self.approach_thresh:
                # Move to edge
                actions[i] = agent_edge_vec
            else:
                # Push the box
                push_vec = package_goal_vec[i]
                press_vec = agent_package_vec[i]  # agent->package
                combined = push_vec + self.pressing_factor * press_vec
                actions[i] = combined
        actions = torch.clamp(actions, min=-u_range, max=u_range)
        return actions
