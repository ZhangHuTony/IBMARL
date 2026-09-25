from src.util.translator import Translator
import numpy as np

class ObsConcatTranslatorPusher(Translator):
    """
    Converts the N-tensor observation at a given step in simulation to a concatenated numpy observation.

    Observation from env (Tensor of size Nx14)
    The 14 components are
    1 - X Position
    2 - Y Position
    3 - Heading
    4 - X Velocity
    5 - Y Velocity
    6 - Angular Velocity
    7 - Agent 0 Rel X.
    8 - Agent 0 Rel Y.
    9 - Agent 1 Rel X.
    10 - Agent 1 Rel Y.
    11 - Agent 2 Rel Y.
    12 - Package to Goal X
    13 - Package to Goal Y
    14 - X Package Rel.
    15 - Y Package Rel.
    16 - Package Velocity X
    17 - Package Velocity Y
    18 - Pacakage Orientation
    19 - Package at Goal (bool)

    Concatenated Observation (Vector of size Nx20)
    5 components from the above vector are agent-agnostic (i.e. they do not depend on the agent observing)
    So we can extract components 5-6, 9-11 from the list above and keep it as global information.
    Then for components 1-4, 7-8, we can concatenate N agents together and tack on the global information to obtain a vector of size (1x(6N + 4)).
    The method 'columns' returns the name of each column in Pandas-form.
    """

    @staticmethod
    def from_(**kwargs):
        """
        Translate Concatenated Form -> Environment Observation.
        :param concat_obs: A Tensor or Numpy Array of size (1x8N)
        :return: A numpy array of size (Nx8)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("concat_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the concat_obs=List")

        # Ensure the correct tensor size
        size = obs.shape
        if size[0] % 20 != 0:
            raise Exception(f"The Observation size must be of size 8N, got size {size}")

        raw_obs = np.reshape(obs, (len(obs) // 20, 20))
        return np.array(raw_obs)

    @staticmethod
    def to_(**kwargs):
        """
        Environment Observation -> Translate Concatenated Form.
        :param raw_obs: A Tensor or Numpy Array of size (Nx20)
        :return: A numpy array of size (1x14N)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("raw_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the raw_obs=[Tensor | Numpy.Array] key")

        # Ensure the correct tensor size
        size = obs.shape
        if size[1] != 20:
            raise Exception(f"The Observation size must be N x 20, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        concat_obs = obs.flatten()
        assert len(concat_obs) == (size[0] * 20)
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (14N) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x",
            "pos_y",
            "pos_heading",
            "vel_x",
            "vel_y",
            "ang_vel",
            "agent_0_rel_x",
            "agent_0_rel_y",
            "agent_1_rel_x",
            "agent_1_rel_y",
            "agent_2_rel_x",
            "agent_2_rel_y",
            "package_to_goal_x",
            "package_to_goal_y",
            "package_rel_x",
            "package_rel_y",
            "package_vel_x",
            "package_vel_y",
            "package_heading",
            "package_at_goal"
        ]
        
        def columns_i(i):
            # Simple Function that returns only the titles for the ith agent
            return [f"A{i}_{AGENT_COLUMNS[j]}" for j in range(len(AGENT_COLUMNS))]

        columns = []
        for i in range(n):
            columns += columns_i(i)
        return columns

    @staticmethod
    def size(n=3):
        raise NotImplementedError("This method is not implemented correctly and should not be used")
        return (6 * n) + 5

    @staticmethod
    def raw_size(n=3):
        return 20 * n

"""
TESTING / EXAMPLE SCRIPT
Note: Must be run as module due to relative imports, please use
`python -m src.scenarios.reverse_transport.util.observation_translator` to run or edit your pycharm config file to module mode
"""
if __name__ == "__main__":
    pass