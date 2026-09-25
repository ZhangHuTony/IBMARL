from ....util.translator import Translator
import numpy as np


class ObsConcatTranslatorBuzzWire(Translator):
    """
    This task is always 2 agents

    Observation from env (Tensor of size 2x6)
    The 6 components are
    1 - X Position
    2 - Y Position
    3 - X Velocity
    4 - Y Velocity
    5 - Rel Goal Position X
    6 - Rel Goal Position Y

    Concatenated Observation (Vector of size 1x12)
    There are no agent-agnostic elements of the observation, so the concatenation is just a flattening of the matrix.

    The method 'columns' returns the name of each column in Pandas-form.
    """

    @staticmethod
    def from_(**kwargs):
        """
        :param concat_obs: A Tensor or Numpy Array of size (1x12)
        :return: A numpy array of size (2x6)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("concat_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the concat_obs=List")

        # Ensure the correct tensor size
        size = obs.shape
        if size[0] % 6 != 0:
            raise Exception(f"The Observation size must be of size 2x6, got size {size}")

        raw_obs = np.reshape(obs, (2, 6))
        return np.array(raw_obs)

    @staticmethod
    def to_(**kwargs):
        """
        :param raw_obs: A Tensor or Numpy Array of size (2x6)
        :return: A numpy array of size (1x12)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("raw_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the raw_obs=[Tensor | Numpy.Array] key")

        # Ensure the correct tensor size
        size = obs.shape
        if size[1] != 6:
            raise Exception(f"The Observation size must be 2 x 6, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        concat_obs = obs.flatten()
        assert len(concat_obs) == (size[0] * 6)
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (6N) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x", "pos_y", "vel_x", "vel_y", "goal_rel_x", "goal_rel_y"
        ]

        def columns_i(i):
            # Simple Function that returns only the titles for the ith agent
            return [f"A{i}_{AGENT_COLUMNS[j]}" for j in range(len(AGENT_COLUMNS))]

        columns = []
        for i in range(n):
            columns += columns_i(i)
        return columns

    @staticmethod
    def size(n=2):
        return n * 6

    @staticmethod
    def raw_size(n=2):
        return n * 6

"""
TESTING / EXAMPLE SCRIPT
`python -m src.scenarios.navigation.util.observation_translator` to run
"""
if __name__ == "__main__":
    # Raw_obs is an example of a 2x6 observation from the environment
    raw_obs = [[-0.3397546708583832, 0.25732308626174927, -0.015873704105615616, -0.009263729676604271, -0.361832857131958, -0.010877043008804321], [0.16089564561843872, 0.2567792534828186, 0.00025500915944576263, -0.0016020596958696842, 0.13881747424602509, -0.011420875787734985]]
    raw_obs = np.array(raw_obs)

    # Convert raw_obs to a concatenated representation
    concat_obs = ObsConcatTranslatorBuzzWire.to_(raw_obs=raw_obs)
    print(concat_obs)

    # Obtain the column labels for the new concatenated features
    columns = ObsConcatTranslatorBuzzWire.columns(n=2)
    print(columns)

    # Perform the Inverse Translation, where we go back to the raw obs from the concatenated version
    raw_inverse = ObsConcatTranslatorBuzzWire.from_(concat_obs=concat_obs)
    print(raw_inverse)