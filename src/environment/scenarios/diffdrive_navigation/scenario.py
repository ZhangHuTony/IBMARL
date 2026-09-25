from vmas.simulator.scenario import BaseScenario
import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.scenarios.navigation import Scenario as NavigationScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    Color,
    DRAG,
    LINEAR_FRICTION,
    ScenarioUtils,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

class DiffDriveNavigation(NavigationScenario):

    def make_world(self, batch_dim, device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 3)
        self.collisions = kwargs.pop("collisions", True)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", False
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        self.goals = []  # We will store our agents' goal entities here for easy access
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )  # Get color for agent

            sensors = [
                Lidar(
                    world,
                    n_rays=self.n_lidar_rays,
                    max_range=self.lidar_range,
                    entity_filter=lambda e: isinstance(
                        e, Agent
                    ),  # This makes sure that this lidar only percieves other agents
                    angle_start=0.0,  # LIDAR angular ranges (we sense 360 degrees)
                    angle_end=2
                    * torch.pi,  # LIDAR angular ranges (we sense 360 degrees)
                )
            ]  # Agent LIDAR sensor

            agent = Agent(
                name=f"diff_drive_{i}",
                collide=True,
                color=color,
                render_action=True,
                sensors=sensors,
                shape=Sphere(radius=self.agent_radius),
                u_range=[1, 1],  # Ranges for actions
                u_multiplier=[0.25, 1],  # Action multipliers
                dynamics=DiffDrive(
                    world
                ),  # If you go to its class you can see it has 2 actions: forward velocity and angular velocity
            )
           
            agent.pos_rew = torch.zeros(
                batch_dim, device=device
            )  # Tensor that will hold the position reward fo the agent
            agent.agent_collision_rew = (
                agent.pos_rew.clone()
            )  # Tensor that will hold the collision reward fo the agent

            world.add_agent(agent)  # Add the agent to the world

            ################
            # Add goals
            ################
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal
            self.goals.append(goal)

        ################
        # Add obstacles
        ################
        # self.obstacles = (
        #     []
        # )  # We will store obstacles here for easy access
        # for i in range(self.n_obstacles):
        #     obstacle = Landmark(
        #         name=f"obstacle_{i}",
        #         collide=True,
        #         color=Color.BLACK,
        #         shape=Sphere(radius=self.agent_radius * 2 / 3),
        #     )
        #     world.add_landmark(obstacle)
        #     self.obstacles.append(obstacle)

        self.pos_rew = torch.zeros(
            batch_dim, device=device
        )  # Tensor that will hold the global position reward
        self.final_rew = (
            self.pos_rew.clone()
        )  # Tensor that will hold the global done reward
        self.all_goal_reached = (
            self.pos_rew.clone()
        )  # Tensor indicating if all goals have been reached

        return world
    
    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_spawning_x, self.world_spawning_x),
            (-self.world_spawning_y, self.world_spawning_y),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        USE_FIXED_GOAL_LOCATIONS = True
        for i in range(len(self.world.agents)):
            if USE_FIXED_GOAL_LOCATIONS:
                if i == 0:
                    position = torch.tensor([[[-0.03, 0.47]] for _ in range(self.world.batch_dim)])
                elif i == 1:
                    position = torch.tensor([[[-0.22, 0.144]] for _ in range(self.world.batch_dim)])
                elif i == 2:
                    position = torch.tensor([[[-0.49, 0.31]] for _ in range(self.world.batch_dim)])
            else:       
                position = ScenarioUtils.find_random_pos_for_entity(
                    occupied_positions=occupied_positions,
                    env_index=env_index,
                    world=self.world,
                    min_dist_between_entities=self.min_distance_between_entities,
                    x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                    y_bounds=(-self.world_spawning_y, self.world_spawning_y),
                )
            goal_poses.append(position.squeeze(1))

            # print("pose_shape: ", position.shape)
            # print("occupied_positions_shape: ", occupied_positions.shape)
            occupied_positions = torch.cat([occupied_positions, position], dim=1)
            # print("Goal locations: ", goal_poses)

        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

    def observation(self, agent: Agent):
        goal_poses = []
        rotation = agent.state.rot % (2 * torch.pi)
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        obs = torch.cat(
            [
                agent.state.pos,
                rotation,
                agent.state.vel,
                agent.state.ang_vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )
        return obs