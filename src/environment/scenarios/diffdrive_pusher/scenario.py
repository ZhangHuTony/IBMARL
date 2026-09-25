import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 3)
        self.n_packages = kwargs.pop("n_packages", 1)
        self.package_width = kwargs.pop("package_width", 0.60)
        self.package_length = kwargs.pop("package_length", 0.12)
        self.package_mass = kwargs.pop("package_mass", 120)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.shaping_factor = 100
        self.world_semidim = 1
        self.agent_radius = 0.07

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
            y_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
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
            (max(n_agents - len(known_colors), 0), 3), device=device
        )

        # Add agents
        for i in range(n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            sensors = [
                Lidar(
                    world,
                    n_rays=1,
                    max_range=0.15,
                    entity_filter=lambda e: isinstance(
                        e, Agent
                    ),  # This makes sure that this lidar only percieves other agents
                    angle_start=0.0,  # LIDAR angular ranges (we sense 360 degrees)
                    angle_end=2
                    * torch.pi,  # LIDAR angular ranges (we sense 360 degrees)
                )
            ]  # Agent LIDAR sensor

            agent = Agent(
                name=f"agent_{i}",
                color=color,
                collide=True,
                shape=Sphere(self.agent_radius),
                sensors=sensors,
                render_action=True,
                u_range=[1, 1],  # Ranges for actions
                u_multiplier=[0.25, 1],  # Action multipliers
                dynamics=DiffDrive(
                    world
                ),  # If you go to its class you can see it has 2 actions: forward velocity and angular velocity
            
            )
            world.add_agent(agent)

        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Box(length=0.1, width=0.4),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.packages = []
        
        package = Landmark(
            name=f"package {i}",
            rotatable=True,
            collide=True,
            movable=True,
            mass=120,
            shape=Box(length=self.package_length, width=self.package_width),
            color=Color.RED,
        )
        package.goal = goal
        self.packages.append(package)
        world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):

        for i, agent in enumerate(self.world.agents):
            agent.state.pos = torch.tensor([[-0.6, -0.3 + i * 0.3] for _ in range(self.world.batch_dim)], device=self.world.device)

        agent_occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)

        goal = self.world.landmarks[0]
        goal.state.pos = torch.tensor([[0.98, 0] for _ in range(self.world.batch_dim)], device=self.world.device)

        ScenarioUtils.spawn_entities_randomly(
            self.packages,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(
                -0.1,
                0.1,
            ),
            y_bounds=(
                -self.world_semidim / 2,
                self.world_semidim / 2,
            ),
            occupied_positions=agent_occupied_positions,
        )

        for package in self.packages:
            package.on_goal = self.world.is_overlapping(package, package.goal)

            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            ) 
            # positions = torch.stack([a.state.pos[:, 0] for a in self.world.agents])
            # if len(positions.shape) == 1:
            #     positions = positions.unsqueeze(0)
            # # print(positions)
            # x_reward_mult = torch.sum(positions, dim=0) * 0.0001
            # # print(x_reward_mult)
            # self.rew *= x_reward_mult

            # print(sum([a.state.pos[0] for a in self.world.agents]))
            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value,
                    device=self.world.device,
                    dtype=torch.float32,
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value,
                    device=self.world.device,
                    dtype=torch.float32,
                )

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        package_obs = []
        package = self.packages[0]

        package_obs.append(package.state.pos - package.goal.state.pos)
        package_obs.append(package.state.pos - agent.state.pos)
        package_obs.append(package.state.vel)
        package_obs.append(package.state.rot % (2 * torch.pi))
        package_obs.append(package.on_goal.unsqueeze(-1))

        rotation = agent.state.rot % (2 * torch.pi)
        return torch.cat(
            [
                agent.state.pos,
                rotation,
                agent.state.vel,
                agent.state.ang_vel,
                self.world.agents[0].state.pos - agent.state.pos,
                self.world.agents[1].state.pos - agent.state.pos,
                self.world.agents[2].state.pos - agent.state.pos,
                *package_obs,
            ],
            dim=-1,
        )

    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )
    
    # def process_action(self, agent: Agent):
    #     old_action = agent.action.u
    #     if -0.06 < old_action[0][0] < 0.06:
    #         agent.action.u[0][0] = 0.0
    #     if -0.60 < old_action[0][1] < 0.60:
    #         agent.action.u[0][1] = 0.0
    #     return agent.action.u
