from ....util.translator import Translator
import numpy as np


class ObsConcatTranslatorBalance(Translator):
    """
    Converts the N-tensor observation at a given step in simulation to a concatenated numpy observation.

    Observation from env (Tensor of size Nx16)
    The 16 components are
    1 - X Position
    2 - Y Position
    3 - X Velocity
    4 - Y Velocity
    5 - X Package Rel.
    6 - Y Package Rel.
    7 - X Line Rel.
    8 - Y Line Rel.
    9 - Package to Goal X
    10 - Package to Goal Y
    11 - Package Velocity X
    12 - Package Velocity Y
    13 - Line Velocity X
    14 - Line Velocity Y
    15 - Line Angular Velocity
    16 - Line Rotation (mod pi)

    Concatenated Observation (Vector of size 1x(8N + 8))
    8 components from the above vector are agent-agnostic (i.e. they do not depend on the agent observing)
    So we can extract components 9-16 from the list above and keep it as global information.
    Then for components 1-8, we can concatenate N agents together and tack on the global information to obtain a vector of size (1x(8N + 8)).
    The method 'columns' returns the name of each column in Pandas-form.
    """

    @staticmethod
    def from_(**kwargs):
        """
        :param concat_obs: A Tensor or Numpy Array of size (1x(8N + 8))
        :return: A numpy array of size (Nx16)
        """
        # Extract the Keyword Args, raise exception if None
        concat_obs = kwargs.get("concat_obs", None)
        if concat_obs is None:
            raise Exception("The Observation Translator must be provided the concat_obs=List")

        # Ensure the correct tensor size
        size = concat_obs.shape
        if size[0] % 8 != 0:
            raise Exception(f"The Observation size must be divisible by 8, got size {size}")

        # Build the N*16 matrix
        global_information = list(concat_obs[-8:])
        output = []
        for i in range((len(concat_obs) // 8) - 1):
            agent_obs = list(concat_obs[8 * i:8 * (i + 1)])
            obs = agent_obs + global_information
            output.append(obs)
        return np.array(output)

    @staticmethod
    def to_(**kwargs):
        """
        :param raw_obs: A Tensor or Numpy Array of size (Nx16)
        :return: A numpy array of size (1x(8N + 8))
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("raw_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the raw_obs=[Tensor | Numpy.Array] key")

        # Ensure the correct tensor size
        size = obs.shape
        if size[1] != 16:
            raise Exception(f"The Observation size must be N x 16, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        global_information = list(obs[0][8:])
        agent_wise_information = [list(o[0:8]) for o in obs]
        agent_wise_information.append(global_information)
        concat_obs = np.array(agent_wise_information)
        concat_obs = concat_obs.flatten()

        assert len(concat_obs) == (size[0] * 8) + 8
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (8N + 8) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x", "pos_y", "vel_x", "vel_y", "package_rel_x", "package_rel_y", "line_rel_x", "line_rel_y"
        ]
        GLOBAL_COLUMNS = [
            "package_to_goal_x", "package_to_goal_y", "package_vel_x", "package_vel_y", "line_vel_x", "line_vel_y", "line_angular_vel", "line_angle"
        ]

        def columns_i(i):
            # Simple Function that returns only the titles for the ith agent
            return [f"A{i}_{AGENT_COLUMNS[j]}" for j in range(len(AGENT_COLUMNS))]

        columns = []
        for i in range(n):
            columns += columns_i(i)
        columns += GLOBAL_COLUMNS
        return columns

    @staticmethod
    def size(n=3):
        return (8 * n) + 8

    @staticmethod
    def raw_size(n=3):
        return 16 * n

"""
TESTING / EXAMPLE SCRIPT
Note: Must be run as module due to relative imports, please use
`python -m src.scenarios.balance.util.observation_translator` to run or edit your pycharm config file to module mode
"""
if __name__ == "__main__":
    # Raw_obs is an example of a 3x16 observation from the environment
    raw_obs = [[-0.8788270950317383, -0.993831992149353, 0.0, 0.06167991831898689, -0.12742137908935547, -0.10466557741165161, -0.38499999046325684, -0.05199843645095825, -0.7439188957214355, -1.6573882102966309, 0.0, 0.008335854858160019, 0.0, -0.0183358546346426, 0.06440682709217072, 0.006440682802349329], [-0.49382710456848145, -0.993831992149353, 0.0, 0.06167991831898689, 0.25757861137390137, -0.10466557741165161, 0.0, -0.05199843645095825, -0.7439188957214355, -1.6573882102966309, 0.0, 0.008335854858160019, 0.0, -0.0183358546346426, 0.06440682709217072, 0.006440682802349329], [-0.10882711410522461, -0.993831992149353, 0.0, 0.06167991831898689, 0.6425786018371582, -0.10466557741165161, 0.38499999046325684, -0.05199843645095825, -0.7439188957214355, -1.6573882102966309, 0.0, 0.008335854858160019, 0.0, -0.0183358546346426, 0.06440682709217072, 0.006440682802349329]]
    raw_obs = np.array(raw_obs)

    # Convert raw_obs to a concatenated representation
    concat_obs = ObsConcatTranslatorBalance.to_(raw_obs=raw_obs)
    print(concat_obs)

    # Obtain the column labels for the new concatenated features
    columns = ObsConcatTranslatorBalance.columns(n=3)
    print(columns)

    # Perform the Inverse Translation, where we go back to the raw obs from the concatenated version
    raw_inverse = ObsConcatTranslatorBalance.from_(concat_obs=concat_obs)
    print(raw_inverse)