import numpy as np
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy

class FootballHeuristicPolicy(BaseHeuristicPolicy):

    def __init__(self, num_blue_agents=3, continuous_actions=True):
        super().__init__(continuous_action=continuous_actions)
        self.num_blue_agents = num_blue_agents

        # Storage for single-step data
        self.current_step_obs = {}
        self.current_step_actions = {}
        self.call_count = 0
        self.current_step = 0

    def compute_action(self, obs_or_obslist, u_range=None, agent_idx=None):

        # 1) Convert the single observation to a flat 8D array
        if isinstance(obs_or_obslist, (list, np.ndarray)) and len(obs_or_obslist) == 1:
            obs = np.array(obs_or_obslist[0], dtype=np.float32)
        else:
            obs = np.array(obs_or_obslist, dtype=np.float32)

        if agent_idx is None:
            agent_idx = self.call_count

        # 2) Store this agent’s obs
        self.current_step_obs[agent_idx] = obs

        # 3) Increment call_count
        self.call_count += 1

        if self.call_count == self.num_blue_agents:

            all_agent_obs = [self.current_step_obs[i] for i in range(self.num_blue_agents)]
            actions = self._compute_actions_multiagent(all_agent_obs, u_range)

            for i, act in enumerate(actions):
                self.current_step_actions[i] = act

            self.call_count = 0
            self.current_step_obs = {}
            self.current_step += 1

        if agent_idx in self.current_step_actions:
            return self.current_step_actions[agent_idx]
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    def _compute_actions_multiagent(self, obs_list, u_range):
        num_agents = len(obs_list)
        agent_positions = []

        # 1) Parse positions
        for obs in obs_list:
            obs = np.array(obs).squeeze()
            if obs.shape != (8,):
                raise ValueError(f"Invalid shape {obs.shape}, expected (8,). obs={obs}")
            my_x, my_y = obs[0], obs[1]
            ball_dx, ball_dy = obs[4], obs[5]
            ball_x, ball_y = my_x + ball_dx, my_y + ball_dy
            agent_positions.append((my_x, my_y, ball_x, ball_y))

        # 2) Determine ball side
        first_ball_x = agent_positions[0][2]
        ball_in_blue_half = (first_ball_x < 0)

        # 3) Distances
        distances = []
        for (my_x, my_y, bx, by) in agent_positions:
            dist = np.linalg.norm([bx - my_x, by - my_y])
            distances.append(dist)
        sorted_indices = np.argsort(distances)
        actions = []
        if ball_in_blue_half:
            # single closest agent
            closest_idx = sorted_indices[0]
            for i in range(num_agents):
                my_x, my_y, ball_x, ball_y = agent_positions[i]
                act = np.array([0.0, 0.0], dtype=np.float32)
                if i == closest_idx:
                    # Attack => move right
                    act[0] = 1.0
                    # a bit of aiming for center
                    act[1] = -0.5 * my_y
                else:
                    # defend => get behind the ball
                    if my_x > ball_x:
                        act[0] = -1.0
                    act[1] = (ball_y - my_y)
                if u_range is not None:
                    act = np.clip(act, -u_range, u_range)
                actions.append(act)
        else:
            # ball in red half => two closest attack
            c1, c2 = sorted_indices[0], sorted_indices[1] if num_agents > 1 else sorted_indices[0]
            for i in range(num_agents):
                my_x, my_y, ball_x, ball_y = agent_positions[i]
                act = np.array([0.0, 0.0], dtype=np.float32)
                if i == c1 or i == c2:
                    # Attack => push to right
                    act[0] = 1.0
                    act[1] = -0.5 * my_y
                else:
                    # Defend => move left if x> -0.3
                    if my_x > -0.3:
                        act[0] = -1.0
                    act[1] = 0.2 * (ball_y - my_y)
                if u_range is not None:
                    act = np.clip(act, -u_range, u_range)
                actions.append(act)

        return actions
