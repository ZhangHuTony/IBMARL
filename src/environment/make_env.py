from torchrl.envs import VmasEnv, TransformedEnv, check_env_specs
from torchrl.envs.transforms import RewardSum, StepCounter
from src.environment.transforms.registry import build_transforms
from src.environment.scenarios.balance_sparse import SparseRewardBalanceScenario
from src.environment.scenarios.buzz_wire_sparse import SparseRewardBuzzWireScenario
from src.environment.scenarios.navigation_terminal import TerminalSuccessNavigationScenario

def make_env(config: dict, device) -> TransformedEnv:
    """
    Create and return a multi-agent environment wrapped with necessary transforms.
    """
    horizon = config.get("horizon", 100)
    num_vmas_env = config.get("frames_per_batch", 1000) // horizon
    # num_vmas_env = 20
    seed = config.get("seed", 42)
    binary = binary_terminal(config)

    # Create the base Vmas environment.
    # Under the binary terminal schema the time limit is NOT given to VMAS:
    # vmas merges its own truncation into scenario.done(), and torchrl's VmasEnv
    # copies that merged flag into "terminated" (it never writes "truncated"),
    # so every horizon hit would look like a true absorbing terminal and the
    # critic would learn V = 0 one step short of the goal.  The StepCounter
    # below produces the time limit instead, as "truncated", leaving
    # "terminated" to the scenario's real terminal.
    base_env = VmasEnv(
        scenario=resolve_scenario(config),
        num_envs=num_vmas_env,
        max_steps=None if binary else horizon,
        device=device,
        seed=seed,
        continuous_actions=True,
        **scenario_kwargs(config),
    )

    # Wrap with TransformedEnv
    env = TransformedEnv(base_env)

    if binary:
        # Root-level done keys only (VmasEnv exposes none per group), so the
        # default keys are right; order relative to RewardSum is immaterial.
        env.append_transform(StepCounter(max_steps=horizon))

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

def binary_terminal(config: dict) -> bool:
    """
    Whether this env uses the binary terminal reward schema: +1 once, on the
    step the task's success predicate first holds, termination there, and a
    bootstrapped time limit otherwise.  Set per task in
    config/environments/<task>.yaml; it is part of the task definition, so
    every learner on that task sees the same MDP.  Only meaningful on top of
    sparse rewards -- ``--dense`` switches it off along with them.
    """
    return bool(config.get("sparse_rewards", False)) and bool(
        config.get("binary_terminal_reward", False)
    )


def resolve_scenario(config: dict):
    """
    Return what VmasEnv should build: usually the scenario name string, but a
    scenario INSTANCE (torchrl forwards it to vmas.make_env untouched) where
    the reward has to live at scenario level:

    * sparse buzz_wire, always -- its predicate needs the ball's position,
      which is not in the observation, so it cannot be a TorchRL transform
      like the other scenarios' sparse rewards
      (see src/environment/scenarios/buzz_wire_sparse.py);
    * navigation under the binary terminal schema -- a transform runs after
      the simulator has decided done(), so it cannot end the episode where the
      reward is paid (see src/environment/scenarios/navigation_terminal.py).

    A fresh instance per call is essential: the train, eval, and render envs
    each need their own simulator state.
    """
    scenario = config.get("scenario_name")
    sparse = config.get("sparse_rewards", False)
    binary = binary_terminal(config)
    if scenario == "buzz_wire" and sparse:
        mode = "binary terminal reward" if binary else "legacy -1/step reward + done suppression"
        print(f"Using SparseRewardBuzzWireScenario ({mode})")
        return SparseRewardBuzzWireScenario(
            success_threshold=config.get("gt_radius", 0.1),
            binary_terminal=binary,
        )
    if scenario == "navigation" and binary:
        print("Using TerminalSuccessNavigationScenario (per-agent binary terminal reward)")
        return TerminalSuccessNavigationScenario()
    if scenario == "balance" and sparse:
        print("Using SparseRewardBalanceScenario (native on-goal predicate)")
        return SparseRewardBalanceScenario()
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
