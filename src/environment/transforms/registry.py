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
    # Balance and buzz_wire use scenario-level sparse rewards because their
    # exact native goal predicates require simulator state.

    if scenario == "navigation" and sparse_rewards:
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
