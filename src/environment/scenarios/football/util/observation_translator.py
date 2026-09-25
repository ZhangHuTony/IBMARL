from ....util.translator import Translator
import numpy as np


class ObsConcatTranslatorFootball(Translator):
    """
    Converts the N-tensor observation at a given step in simulation to a concatenated numpy observation.

    Observation from env (Tensor of size Nx8)
    The 8 components are
    1 - X Position
    2 - Y Position
    3 - X Velocity
    4 - Y Velocity
    5 - Package Velocity X
    6 - Package Velocity Y
    7 - X Package Rel.
    8 - Y Package Rel.

    Concatenated Observation (Vector of size 1x8N)
    There are no agent-agnostic elements of the observation, so the concatenation is just a flattening of the matrix.

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
        if size[0] % 8 != 0:
            raise Exception(f"The Observation size must be of size 8N, got size {size}")

        raw_obs = np.reshape(obs, (len(obs) // 8, 8))
        return np.array(raw_obs)

    @staticmethod
    def to_(**kwargs):
        """
        Environment Observation -> Translate Concatenated Form.
        :param raw_obs: A Tensor or Numpy Array of size (Nx8)
        :return: A numpy array of size (1x8N)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("raw_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the raw_obs=[Tensor | Numpy.Array] key")

        # Ensure the correct tensor size
        size = obs.shape
        if size[1] != 8:
            raise Exception(f"The Observation size must be N x 8, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        concat_obs = obs.flatten()
        assert len(concat_obs) == (size[0] * 8)
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (8N) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x", "pos_y", "vel_x", "vel_y", "ball_rel_pos_x", "ball_rel_pos_y", "ball_rel_vel_x", "ball_rel_vel_y"
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
        return n * 8

"""
TESTING / EXAMPLE SCRIPT
`python -m src.scenarios.football.util.observation_translator`
"""
if __name__ == "__main__":
    # Raw_obs is an example of a 3x8 observation from the environment
    raw_obs = [[-0.7556151151657104, 0.3548126518726349, 0.0, -0.08025261014699936, 0.7556151151657104, -0.3548126518726349, 0.0, 0.08025261014699936], [-1.3277888298034668, -0.572114109992981, 0.07395011931657791, -0.020968496799468994, 1.3277888298034668, 0.572114109992981, -0.07395011931657791, 0.020968496799468994], [-1.0388658046722412, 0.20111799240112305, 0.0, 0.0, 1.0388658046722412, -0.20111799240112305, 0.0, 0.0]]
    raw_obs = np.array(raw_obs)
    print(raw_obs.shape)

    # Convert raw_obs to a concatenated representation
    concat_obs = ObsConcatTranslatorFootball.to_(raw_obs=raw_obs)
    print(concat_obs)

    # Obtain the column labels for the new concatenated features
    columns = ObsConcatTranslatorFootball.columns(n=3)
    print(columns)

    # Perform the Inverse Translation, where we go back to the raw obs from the concatenated version
    raw_inverse = ObsConcatTranslatorFootball.from_(concat_obs=concat_obs)
    print(raw_inverse)