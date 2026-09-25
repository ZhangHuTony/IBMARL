from mpmath.libmp import from_float

from ....util.translator import Translator
import numpy as np
import torch


class InfoConcatTranslatorDiffDriveNavigation(Translator):
    """
    Converts the info dictionary at a given step in simulation to a concatenated numpy array.

    Info from env for each agent (Dictionary with 2 Keys), e.g.
    "agent_0" : {
        'pos_rew' : tensor(-0.001),
        'final_rew' : tensor(-10.0),
        'agent_collisions' : tensor(0.0),
    },
    ...

    "ground_rew" is the reward penalty for hitting the ground
    "pos_rew" is the reward that encourages progress toward the goal.

    Concatenated Observation (Vector of size 1x2N), where "pos_rew", "ground_rew" are stored for each agent
    The method 'columns' returns the name of each column in Pandas-form.
    """

    @staticmethod
    def from_(**kwargs):
        """
        Translate Concatenated Form -> Environment Info.
        :param concat_info: A Tensor or Numpy Array of size (1x2N)
        :return: An info dictionary with N keys.
        """
        # Extract the Keyword Args, raise exception if None
        info = kwargs.get("concat_info", None)
        if info is None:
            raise Exception("The Info Translator must be provided the concat_info=List")

        info = np.reshape(np.array(info), (len(info) // 3, 3))
        keys = ['pos_rew', 'final_rew', 'agent_collisions']
        info_dict = {f"agent_{i}": {keys[j]: torch.tensor(info[i][j]) for j in range(len(keys))} for i in range(info.shape[0])}
        return info_dict

    @staticmethod
    def to_(**kwargs):
        """
        Translate Environment Info -> Concatenated Form.
        :param raw_info: A Dictionary of info for timestep t, containing N keys.
        :return: A numpy array of size (1x2N)
        """
        # Extract the Keyword Args, raise exception if None
        info = kwargs.get("raw_info", None)
        if info is None:
            raise Exception("The Info Translator must be provided the raw_info=Dict key")

        out = []
        for i in range(len(info.keys())):
            pos_r = info[f"agent_{i}"]["pos_rew"].item()  # Convert from Tensor to Scalar
            final_r = info[f"agent_{i}"]["final_rew"].item()  # Convert from Tensor to Scalar
            collision_r = info[f"agent_{i}"]["agent_collisions"].item()  # Convert from Tensor to Scalar
            out.append(pos_r)
            out.append(final_r)
            out.append(collision_r)

        info_concat = np.array(out)
        return info_concat

    @staticmethod
    def columns(n=3):
        """
        A helper function that creates the numpy Columns for the concatenated observation.
        :param n: Number of Agents
        :return: A list of size (8N + 8) that contains the string titles in-order for the reduced observation space.
        """
        return [f"A{i}_{d}" for i in range(n) for d in ['pos_rew', 'final_rew', 'agent_collisions']]


"""
TESTING / EXAMPLE SCRIPT
Note: Must be run as module due to relative imports, please use
`python -m src.scenarios.navigation.util.info_translator` to run or edit your pycharm config file to module mode
"""
if __name__ == "__main__":
    # Raw Info is an example of a dictionary of len=3 (info) from the environment
    raw_info = {'agent_0': {'pos_rew': torch.tensor(0.0349), 'final_rew': torch.tensor(0.), 'agent_collisions': torch.tensor(0.)}, 'agent_1': {'pos_rew': torch.tensor(0.0349), 'final_rew': torch.tensor(0.), 'agent_collisions': torch.tensor(0.)}, 'agent_2': {'pos_rew': torch.tensor(0.0349), 'final_rew': torch.tensor(0.), 'agent_collisions': torch.tensor(0.)}}
    
    # Convert raw_info to a concatenated representation
    concat_info = InfoConcatTranslatorNavigation.to_(raw_info=raw_info)
    print(concat_info)

    # Obtain the column labels for the new concatenated features
    columns = InfoConcatTranslatorNavigation.columns(n=3)
    print(columns)

    # Perform the Inverse Translation, where we go back to the raw info from the concatenated version
    raw_inverse = InfoConcatTranslatorNavigation.from_(concat_info=concat_info)
    print(raw_inverse)