'''
Registry for custom environment transforms.
'''

from src.environment.transforms.navigation_sparse import NavigationSparseReward

def build_transforms(config: dict):
    """
    Build and return a list of environment transforms based on the configuration.
    """
    scenario = config.get("scenario_name")
    sparse_rewards = config.get("sparse_rewards", False)

    transforms = []

    if scenario == "navigation" and sparse_rewards:
        transforms.append(
            NavigationSparseReward(
                group="agents",
                success_threshold=config.get("success_threshold", 0.5),
                # Ensure this matches your VMAS scenario observation layout
                rel_goal_slice=slice(4, 6), 
            )
        )
    
    return transforms    
