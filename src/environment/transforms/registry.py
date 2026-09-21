'''
Registry for custom environment transforms.
'''

from src.environment.transforms.navigation_sparse import NavigationSparseReward
from src.environment.transforms.transport_sparse import TransportSparseReward

def build_transforms(config: dict):
    """
    Build and return a list of environment transforms based on the configuration.
    """
    scenario = config.get("scenario_name")
    sparse_rewards = config.get("sparse_rewards", False)

    transforms = []
    transform_repr = []

    # NOTE: buzz_wire's sparse reward is NOT a transform - the ball position it
    # needs is absent from the observation, so it lives at scenario level
    # (src/environment/scenarios/buzz_wire_sparse.py via resolve_scenario in
    # make_env.py); buzz_wire intentionally has no branch here.  Navigation
    # under binary_terminal_reward likewise moves to scenario level
    # (scenarios/navigation_terminal.py): a transform cannot make the episode
    # end where the reward is paid.  This transform is its legacy -1/step
    # variant, still selected when the flag is off.
    binary = bool(config.get("binary_terminal_reward", False))

    if scenario == "navigation" and sparse_rewards and not binary:
        transform_repr.append("NavigationSparseReward")
        transforms.append(
            NavigationSparseReward(
                group="agents",
                success_threshold=config.get("gt_radius"),
                # Ensure this matches your VMAS scenario observation layout
                rel_goal_slice=slice(4, 6), 
            )
        )
    elif scenario == "transport" and sparse_rewards:
        transforms.append(
            TransportSparseReward(
                group="agents",
                # Success is the package's on_goal flag, not a distance
                # threshold, so there is no gt_radius to pass here.
                n_packages=config.get("n_packages", 1),
            )
        )
    return transforms
