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

    if scenario == "navigation" and sparse_rewards:
        
        return [NavigationSparseReward(
                group="agents",
                success_threshold=config["reward"].get("success_threshold", 0.05),
                # ONE of these must be correct for navigation:
                rel_goal_slice=slice(0, 2),  # change if needed
                # OR:
                # distance_index=some_index
            )]
    
    else:
        return []
                                      
