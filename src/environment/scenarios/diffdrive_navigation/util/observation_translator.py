from ....util.translator import Translator
import numpy as np


class ObsConcatTranslatorDiffDriveNavigation(Translator):
    """
    Converts the N-tensor observation at a given step in simulation to a concatenated numpy observation.

    Observation from env (Tensor of size Nx18)
    The 20 components are
    1 - X Position
    2 - Y Position
    3 - orientation
    4 - X Velocity
    5 - Y Velocity
    6 - Angular Velocity
    7 - Rel Goal Position X
    8 - Rel Goal Position Y
    9-20 - LIDAR Sensors

    Concatenated Observation (Vector of size 1x(18N))
    There are no agent-agnostic elements of the observation, so the concatenation is just a flattening of the matrix.

    The method 'columns' returns the name of each column in Pandas-form.
    """

    @staticmethod
    def from_(**kwargs):
        """
        :param concat_obs: A Tensor or Numpy Array of size (1x(18N))
        :return: A numpy array of size (Nx18)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("concat_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the concat_obs=List")

        # Ensure the correct tensor size
        size = obs.shape
        if size[0] % 20 != 0:
            raise Exception(f"The Observation size must be of size Nx18, got size {size}")

        print(obs)
        raw_obs = np.reshape(obs, (len(obs) // 20, 20))
        return np.array(raw_obs)

    @staticmethod
    def to_(**kwargs):
        """
        :param raw_obs: A Tensor or Numpy Array of size (Nx18)
        :return: A numpy array of size (1x(18N))
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("raw_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the raw_obs=[Tensor | Numpy.Array] key")

        # Ensure the correct tensor size
        size = obs.shape
        if size[1] != 20:
            raise Exception(f"The Observation size must be N x 18, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        concat_obs = obs.flatten()
        assert len(concat_obs) == (size[0] * 20)
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (18N) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x", "pos_y", "heading", "vel_x", "vel_y", "ang_vel", "goal_rel_x", "goal_rel_y"
        ]
        LIDAR_COLUMNS = [
            f"lidar_{i}" for i in range(12)
        ]
        AGENT_COLUMNS += LIDAR_COLUMNS

        def columns_i(i):
            # Simple Function that returns only the titles for the ith agent
            return [f"A{i}_{AGENT_COLUMNS[j]}" for j in range(len(AGENT_COLUMNS))]

        columns = []
        for i in range(n):
            columns += columns_i(i)
        return columns

    @staticmethod
    def size(n=3):
        return 20 * n

    @staticmethod
    def raw_size(n=3):
        return 20 * n

"""
TESTING / EXAMPLE SCRIPT
`python -m src.scenarios.navigation.util.observation_translator` to run
"""
if __name__ == "__main__":
    # Raw_obs is an example of a 3x16 observation from the environment
    raw_obs = [[-0.007873652502894402, 0.6864399909973145, 0.0012899432331323624, 1.3184422641643323e-05, 0.011939527466893196, -0.10644948482513428, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.11032746732234955, 0.276554137468338, -0.0007248725742101669, 2.3229895305121318e-05, -0.021583423018455505, 0.011941581964492798, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1748199611902237, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.3851543664932251, 0.26815736293792725, 0.0, 0.0, -0.08294129371643066, 0.46472275257110596, 0.1748199611902237, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    raw_obs = np.array(raw_obs)
    print(raw_obs.shape)

    # Convert raw_obs to a concatenated representation
    concat_obs = ObsConcatTranslatorNavigation.to_(raw_obs=raw_obs)
    print(concat_obs)

    # Obtain the column labels for the new concatenated features
    columns = ObsConcatTranslatorNavigation.columns(n=3)
    print(columns)

    # Perform the Inverse Translation, where we go back to the raw obs from the concatenated version
    raw_inverse = ObsConcatTranslatorNavigation.from_(concat_obs=concat_obs)
    print(raw_inverse)