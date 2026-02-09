"""
Load a policy checkpoint from an experiment folder and render VMAS rollouts.
Runs N seeded rollouts and saves a video for each.
"""

from pathlib import Path

import torch
import yaml
from tensordict.nn import TensorDictSequential
from torchrl.envs import ExplorationType, TransformedEnv, set_exploration_type
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

from src.environment.make_env import make_env
from src.experiments.ibmarl.networks import build_rl_policies

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "/home/connor/Desktop/Projects/IBMARL/results/ibmarl_navigation_2026-02-09_09-35-57"
N_SEEDS = 3
HORIZON = 100  # Steps per rollout
# -----------------------------------------------------------------------------


def load_experiment_config(exp_dir: Path) -> dict:
    """Load merged config from experiment folder."""
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_checkpoint(exp_dir: Path) -> dict:
    """Load policy checkpoint (dict of group -> state_dict)."""
    ckpt_path = exp_dir / "checkpoints" / "policy_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)


def build_policies_from_checkpoint(config: dict, env, device: torch.device):
    """
    Build policy architecture and load weights from checkpoint.
    Uses the same structure as MADDPG/IBMARL RL policies.
    """
    policies, _ = build_rl_policies(config, env, device)
    return policies


def render_rollout(
    base_env,
    policy: TensorDictSequential,
    videos_dir: Path,
    seed: int,
    horizon: int,
) -> None:
    """Run one rollout with rendering and save to video."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    video_logger = CSVLogger(
        exp_name="render_logs",
        log_dir=str(videos_dir),
        video_format="mp4",
    )

    # Clone env for this rollout (avoids mutating shared env)
    env_with_render = TransformedEnv(base_env.base_env, base_env.transform.clone())
    env_with_render = env_with_render.append_transform(
        PixelRenderTransform(
            out_keys=["pixels"],
            preproc=lambda x: x.copy(),
            as_non_tensor=True,
            mode="rgb_array",
        )
    )
    env_with_render = env_with_render.append_transform(
        VideoRecorder(logger=video_logger, tag=f"rollout_seed_{seed}")
    )

    policy.eval()
    with torch.no_grad():
        with set_exploration_type(ExplorationType.MODE):
            env_with_render.rollout(horizon, policy=policy)

    env_with_render.transform.dump()
    print(f"  Saved rollout (seed={seed})")


def main():
    exp_dir = Path(EXPERIMENT_DIR)
    if not exp_dir.is_absolute():
        exp_dir = PROJECT_ROOT / exp_dir

    print(f"Loading experiment from: {exp_dir}")
    config = load_experiment_config(exp_dir)
    checkpoint = load_checkpoint(exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build env and policy (env used for policy structure; reuse for rollouts)
    print("Building environment...")
    render_config = {
        **config,
        "frames_per_batch": HORIZON,  # Single parallel env for clear video
    }
    env = make_env(render_config, device)

    print("Building policy and loading checkpoint...")
    policies = build_policies_from_checkpoint(config, env, device)
    for group, state_dict in checkpoint.items():
        if group in policies:
            policies[group].load_state_dict(state_dict, strict=True)
        else:
            raise KeyError(f"Checkpoint has group '{group}' but env has {list(policies.keys())}")

    policy = TensorDictSequential(*policies.values())

    videos_dir = exp_dir / "rendered_rollouts"
    videos_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRunning {N_SEEDS} rollouts, saving to {videos_dir}")

    base_seed = config.get("seed", 42)
    for i in range(N_SEEDS):
        seed = base_seed + i
        # Create fresh env with this seed for reproducible rollout
        seed_config = {**render_config, "seed": seed}
        env_for_run = make_env(seed_config, device)
        print(f"  Rollout {i + 1}/{N_SEEDS} (seed={seed})...")
        render_rollout(env_for_run, policy, videos_dir, seed, HORIZON)

    print(f"\nDone. Videos saved under: {videos_dir.resolve()}/render_logs/videos/")


if __name__ == "__main__":
    main()
