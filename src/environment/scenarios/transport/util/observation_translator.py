from ....util.translator import Translator
import numpy as np

class ObsConcatTranslatorTransport(Translator):
    """
    Converts the N-tensor observation at a given step in simulation to a concatenated numpy observation.

    Observation from env (Tensor of size Nx11)
    The 11 components are
    1 - X Position
    2 - Y Position
    3 - X Velocity
    4 - Y Velocity
    5 - Package to Goal X
    6 - Package to Goal Y
    7 - X Package Rel.
    8 - Y Package Rel.
    9 - Package Velocity X
    10 - Package Velocity Y
    11 - Package at Goal (bool)

    Concatenated Observation (Vector of size 1x(6N + 5))
    5 components from the above vector are agent-agnostic (i.e. they do not depend on the agent observing)
    So we can extract components 5-6, 9-11 from the list above and keep it as global information.
    Then for components 1-4, 7-8, we can concatenate N agents together and tack on the global information to obtain a vector of size (1x(6N + 4)).
    The method 'columns' returns the name of each column in Pandas-form.
    """

    @staticmethod
    def from_(**kwargs):
        """
        :param concat_obs: A Tensor or Numpy Array of size (1x(6N + 5))
        :return: A numpy array of size (Nx11)
        """
        # Extract the Keyword Args, raise exception if None
        concat_obs = kwargs.get("concat_obs", None)
        if concat_obs is None:
            raise Exception("The Observation Translator must be provided the concat_obs=List")

        # Ensure the correct tensor size
        size = concat_obs.shape
        if (size[0] - 5) % 6 != 0:
            raise Exception(f"The Observation size must be of size (6N + 5), got size {size}")

        # Build the N*10 matrix
        # Note, the ordering of the global and local observation states are really entangled for this environment,
        #  so that is why the index ordering is very weird
        # See VMAS reserve_transport.py at https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/scenarios/reverse_transport.py
        global_information = list(concat_obs[-5:])
        package_vel = global_information[0:2]
        package_to_goal = global_information[2:]
        output = []
        for i in range((len(concat_obs) - 4) // 6):
            agent_obs = list(concat_obs[6 * i:6 * (i + 1)])
            pos_and_vel = agent_obs[0:4]
            rel_package = agent_obs[4:]
            obs = pos_and_vel + package_vel + rel_package + package_to_goal
            output.append(obs)
        return np.array(output)

    @staticmethod
    def to_(**kwargs):
        """
        :param raw_obs: A Tensor or Numpy Array of size (Nx11)
        :return: A numpy array of size (1x(6N + 5))
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("raw_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the raw_obs=[Tensor | Numpy.Array] key")

        # Ensure the correct tensor size
        size = obs.shape
        if size[1] != 11:
            raise Exception(f"The Observation size must be N x 11, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        global_information = [obs[0][4], obs[0][5], obs[0][8], obs[0][9], obs[0][10]]  # See the docstring at the top of this class for why and how this works.
        agent_wise_information = [list(o[0:4]) + list(o[6:8]) for o in obs]
        agent_wise_information.append(global_information)
        concat = []
        for info in agent_wise_information:
            concat.extend(info)
        concat_obs = np.array(concat)
        concat_obs = concat_obs.flatten()

        assert len(concat_obs) == (size[0] * 6) + 5
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (8N + 8) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x", "pos_y", "vel_x", "vel_y", "package_rel_x", "package_rel_y",
        ]
        GLOBAL_COLUMNS = [
            "package_to_goal_x", "package_to_goal_y", "package_vel_x", "package_vel_y", "package_at_goal"
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
        return (6 * n) + 5

    @staticmethod
    def raw_size(n=3):
        return 11 * n

"""
TESTING / EXAMPLE SCRIPT
Note: Must be run as module due to relative imports, please use
`python -m src.scenarios.reverse_transport.util.observation_translator` to run or edit your pycharm config file to module mode
"""
if __name__ == "__main__":
    # Raw_obs is an example of a 3x11 observation from the environment
    raw_obs = [[0.20855866372585297, -0.03953355550765991, 0.23984839022159576, -7.611858018208295e-05, 0.4840792417526245, 0.3354102373123169, 0.1867765337228775, 0.639556348323822, 0.0, 0.0, 0.0], [-0.44624027609825134, -0.5416422486305237, 0.023983202874660492, 0.2323441505432129, 0.4840792417526245, 0.3354102373123169, 0.8415755033493042, 1.141664981842041, 0.0, 0.0, 0.0], [-0.3851543664932251, 0.26815736293792725, 0.0, 0.0, 0.4840792417526245, 0.3354102373123169, 0.7804895639419556, 0.33186542987823486, 0.0, 0.0, 0.0]]
    raw_obs = np.array(raw_obs)

    # Convert raw_obs to a concatenated representation
    concat_obs = ObsConcatTranslatorTransport.to_(raw_obs=raw_obs)
    print(concat_obs)

    # Obtain the column labels for the new concatenated features
    columns = ObsConcatTranslatorTransport.columns(n=3)
    print(columns)

    # Perform the Inverse Translation, where we go back to the raw obs from the concatenated version
    raw_inverse = ObsConcatTranslatorTransport.from_(concat_obs=concat_obs)
    print(raw_inverse)