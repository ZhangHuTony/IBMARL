from torchrl.envs import VmasEnv, TransformedEnv, check_env_specs
from torchrl.envs.transforms import RewardSum
from src.environment.transforms.registry import build_transforms

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
        continuous_actions=True,
        **scenario_kwargs(config),
    )

    # Wrap with TransformedEnv
    env = TransformedEnv(base_env)

    # 1. Add Custom Transforms (e.g., Sparse Rewards)
    # Adding these before RewardSum ensures the sums reflect the modified rewards.
    for tr in build_transforms(config):
        env.append_transform(tr)

    # 2. Add RewardSum to track episode totals
    env.append_transform(
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"] * len(base_env.group_map.keys())
        )
    )
    
    # Verify that the specs match the actual data produced by the env
    print("Verifying environment specs...")
    check_env_specs(env)
    return env

def scenario_kwargs(config: dict) -> dict:
    """
    Extract scenario-specific keyword arguments from the configuration.
    """
    scenario = config.get("scenario_name")
    if scenario == "navigation":
        return {"n_agents": config.get("n_agents", 3)}
    return {}