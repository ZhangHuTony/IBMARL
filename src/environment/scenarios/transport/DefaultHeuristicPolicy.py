import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = 0.0  # evaluate u at this value along the spline
        self.start_vel_dist_from_target_ratio = (
            0.5  # distance away from the target for the start_vel to point
        )
        self.start_vel_behind_ratio = 0.5  # component of start vel pointing directly behind target (other component is normal)
        self.start_vel_mag = 1.0  # magnitude of start_vel (determines speed along the whole trajectory, as spline is recalculated continuously)
        self.hit_vel_mag = 1.0
        self.package_radius = 0.15 / 2
        self.agent_radius = -0.02
        self.dribble_slowdown_dist = 0.0
        self.speed = 0.95

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        # agent_vel = observation[:, 2:4]
        package_pos = observation[:, 6:8] + agent_pos
        goal_pos = -observation[:, 4:6] + package_pos
        # control = self.get_action(goal_pos, curr_pos=agent_pos, curr_vel=agent_vel)
        control = self.dribble(agent_pos, package_pos, goal_pos)
        control *= self.speed * u_range
        return torch.clamp(control, -u_range, u_range)

    def dribble(self, agent_pos, package_pos, goal_pos, agent_vel=None):
        package_disp = goal_pos - package_pos
        ball_dist = package_disp.norm(dim=-1)
        direction = package_disp / ball_dist[:, None]
        hit_pos = package_pos - direction * (self.package_radius + self.agent_radius)
        hit_vel = direction * self.hit_vel_mag
        start_vel = self.get_start_vel(
            hit_pos, hit_vel, agent_pos, self.start_vel_mag * 2
        )
        slowdown_mask = ball_dist <= self.dribble_slowdown_dist
        hit_vel[slowdown_mask, :] *= (
            ball_dist[slowdown_mask, None] / self.dribble_slowdown_dist
        )
        return self.get_action(
            target_pos=hit_pos,
            target_vel=hit_vel,
            curr_pos=agent_pos,
            curr_vel=agent_vel,
            start_vel=start_vel,
        )

    def hermite(self, p0, p1, p0dot, p1dot, u=0.0, deriv=0):
        # Formatting
        u = u.reshape((-1,))

        # Calculation
        U = torch.stack(
            [
                self.nPr(3, deriv) * (u ** max(0, 3 - deriv)),
                self.nPr(2, deriv) * (u ** max(0, 2 - deriv)),
                self.nPr(1, deriv) * (u ** max(0, 1 - deriv)),
                self.nPr(0, deriv) * (u**0),
            ],
            dim=1,
        ).float()
        A = torch.tensor(
            [
                [2.0, -2.0, 1.0, 1.0],
                [-3.0, 3.0, -2.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=U.device,
        )
        P = torch.stack([p0, p1, p0dot, p1dot], dim=1)
        ans = U[:, None, :] @ A[None, :, :] @ P
        ans = ans.squeeze(1)
        return ans

    def nPr(self, n, r):
        if r > n:
            return 0
        ans = 1
        for k in range(n, max(1, n - r), -1):
            ans = ans * k
        return ans

    def get_start_vel(self, pos, vel, start_pos, start_vel_mag):
        start_vel_mag = torch.as_tensor(start_vel_mag, device=self.device).view(-1)
        goal_disp = pos - start_pos
        goal_dist = goal_disp.norm(dim=-1)
        vel_mag = vel.norm(dim=-1)
        vel_dir = vel.clone()
        vel_dir[vel_mag > 0] /= vel_mag[vel_mag > 0, None]
        goal_dir = goal_disp / goal_dist[:, None]

        vel_dir_normal = torch.stack([-vel_dir[:, 1], vel_dir[:, 0]], dim=1)
        dot_prod = (goal_dir * vel_dir_normal).sum(dim=1)
        vel_dir_normal[dot_prod > 0, :] *= -1

        dist_behind_target = self.start_vel_dist_from_target_ratio * goal_dist
        point_dir = -vel_dir * self.start_vel_behind_ratio + vel_dir_normal * (
            1 - self.start_vel_behind_ratio
        )

        target_pos = pos + point_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = target_disp.norm(dim=1)
        start_vel_aug_dir = target_disp
        start_vel_aug_dir[target_dist > 0] /= target_dist[target_dist > 0, None]
        start_vel = start_vel_aug_dir * start_vel_mag[:, None]
        return start_vel

    def get_action(
        self,
        target_pos,
        target_vel=None,
        start_pos=None,
        start_vel=None,
        curr_pos=None,
        curr_vel=None,
    ):
        if curr_pos is None:  # If None, target_pos is assumed to be a relative position
            curr_pos = torch.zeros(target_pos.shape, device=self.device)
        if curr_vel is None:  # If None, curr_vel is assumed to be 0
            curr_vel = torch.zeros(target_pos.shape, device=self.device)
        if (
            start_pos is None
        ):  # If None, start_pos is assumed to be the same as curr_pos
            start_pos = curr_pos
        if target_vel is None:  # If None, target_vel is assumed to be 0
            target_vel = torch.zeros(target_pos.shape, device=self.device)
        if start_vel is None:  # If None, start_vel is calculated with get_start_vel
            start_vel = self.get_start_vel(
                target_pos, target_vel, start_pos, self.start_vel_mag * 2
            )

        u_start = torch.ones(curr_pos.shape[0], device=self.device) * self.lookahead
        des_curr_pos = self.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=u_start,
            deriv=0,
        )
        des_curr_vel = self.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=u_start,
            deriv=1,
        )
        des_curr_pos = torch.as_tensor(des_curr_pos, device=self.device)
        des_curr_vel = torch.as_tensor(des_curr_vel, device=self.device)
        control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (des_curr_vel - curr_vel)
        return control