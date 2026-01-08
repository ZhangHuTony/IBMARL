from torchrl.envs import VmasEnv, TransformedEnv
from torchrl.envs.transforms import RewardSum

from transforms.registry import build_transforms

def make_env(config: dict, device) -> TransformedEnv:
    """
    Create and return a multi-agent environment wrapped with necessary transforms.
    """
    scenario_name = config.get("scenario_name")

    horizon = config.get("horizon", 100)
    num_vmas_env = config.get("frames_per_batch", 1000) // horizon
    seed = config.get("seed", 42)

    
    # Create the base Vmas environment
    base_env = VmasEnv(
        scenario=scenario_name,
        num_envs=num_vmas_env,
        max_steps=horizon,
        device=device,
        seed=seed,
        continuous_actions= True,
        **scenario_kwargs(config),
    )

    # Wrap the environment with transforms  
    env = TransformedEnv(base_env)

    for tr in build_transforms(config):
        env.append_transform(tr)
    
    return env


def scenario_kwargs(config: dict) -> dict:
    """
    Extract scenario-specific keyword arguments from the configuration.
    """
    scenario = config.get("scenario_name")
    
    if scenario == "navigation":
        return {
            "n_agents": config.get("n_agents", 3),
        }
    
    else:
        return {}