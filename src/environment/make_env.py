from torchrl.envs import VmasEnv, TransformedEnv, check_env_specs
from torchrl.envs.transforms import RewardSum
from src.environment.transforms.registry import build_transforms
from src.environment.scenarios.buzz_wire_sparse import SparseRewardBuzzWireScenario

def make_env(config: dict, device) -> TransformedEnv:
    """
    Create and return a multi-agent environment wrapped with necessary transforms.
    """
    horizon = config.get("horizon", 100)
    num_vmas_env = config.get("frames_per_batch", 1000) // horizon
    # num_vmas_env = 20
    seed = config.get("seed", 42)

    # Create the base Vmas environment
    base_env = VmasEnv(
        scenario=resolve_scenario(config),
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

def resolve_scenario(config: dict):
    """
    Return what VmasEnv should build: usually the scenario name string, but for
    sparse buzz_wire a scenario INSTANCE (torchrl forwards it to vmas.make_env
    untouched). The sparse reward there needs the ball's position, which is not
    in the observation, so it cannot be a TorchRL transform like the other
    scenarios' sparse rewards (see src/environment/scenarios/buzz_wire_sparse.py).
    A fresh instance per call is essential: the train, eval, and render envs
    each need their own simulator state.
    """
    scenario = config.get("scenario_name")
    if scenario == "buzz_wire" and config.get("sparse_rewards", False):
        print("Using SparseRewardBuzzWireScenario (scenario-level sparse reward + done suppression)")
        return SparseRewardBuzzWireScenario(
            success_threshold=config.get("gt_radius", 0.1)
        )
    return scenario


def scenario_kwargs(config: dict) -> dict:
    """
    Extract scenario-specific keyword arguments from the configuration.
    """
    scenario = config.get("scenario_name")
    if scenario == "navigation":
        return {"n_agents": config.get("n_agents", 3)}
    if scenario == "transport":
        # VMAS transport defaults to 4 agents; without this the n_agents in
        # config/environments/transport.yaml is silently ignored.  Only keys
        # the scenario actually pops may be passed -- make_world ends with
        # ScenarioUtils.check_kwargs_consumed(kwargs) and raises otherwise.
        kwargs = {"n_agents": config.get("n_agents", 3)}
        if "n_packages" in config:
            kwargs["n_packages"] = config["n_packages"]
        return kwargs
    return {}