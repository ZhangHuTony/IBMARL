from ....util.translator import Translator
import numpy as np

class ObsConcatTranslatorWheel(Translator):
    """
    Converts the N-tensor observation at a given step in simulation to a concatenated numpy observation.

    Observation from env (Tensor of size Nx9)
    The 11 components are
    1 - X Position
    2 - Y Position
    3 - X Velocity
    4 - Y Velocity
    5 - Line to Agent X
    6 - Line to Agent Y
    7 - Endpoint1 to Agent X
    8 - Endpoint1 to Agent Y
    9 - Endpoint2 to Agent X
    10 - Endpoint2 to Agent Y
    11 - Line Rotation
    12 - Line Angular Vel
    13 - Difference Between Goal Vel and Current Vel
    """

    @staticmethod
    def from_(**kwargs):
        """
        :param concat_obs: A Tensor or Numpy Array of size (1x(6N + 5))
        :return: A numpy array of size (Nx9)
        """
        # Extract the Keyword Args, raise exception if None
        obs = kwargs.get("concat_obs", None)
        if obs is None:
            raise Exception("The Observation Translator must be provided the concat_obs=List")

        # Ensure the correct tensor size
        size = obs.shape
        if size[0] % 13 != 0:
            raise Exception(f"The Observation size must be of size 9N, got size {size}")

        raw_obs = np.reshape(obs, (len(obs) // 13, 13))
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
        if size[1] != 13:
            raise Exception(f"The Observation size must be N x 9, got size {size}")

        # Translate the Raw Observation into a Concatenated Form
        concat_obs = obs.flatten()
        assert len(concat_obs) == (size[0] * 13)
        return concat_obs

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (N x 13) that contains the string titles in-order for the reduced observation space.
        """
        AGENT_COLUMNS = [
            "pos_x", "pos_y", "vel_x", "vel_y", "line_rel_x", "line_rel_y", "EP1_rel_x", "EP1_rel_y", "EP2_rel_x", "EP2_rel_y", "rot_line", "line_omega", "omega_minus_desired"
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
        return (13 * n)

"""
TESTING / EXAMPLE SCRIPT
Note: Must be run as module due to relative imports, please use
`python -m src.scenarios.reverse_transport.util.observation_translator` to run or edit your pycharm config file to module mode
"""
if __name__ == "__main__":
    pass