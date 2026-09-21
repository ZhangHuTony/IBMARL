"""Train an R2BC demonstrator and export IBMARL-compatible artifacts.

This is the decentralized ``r2bc_decent`` path from the companion R2BC
repository, packaged here so IBMARL can generate its own suboptimal teachers.
The collection algorithm is intentionally unchanged: for each controlled agent
the supervisor supplies only that agent's action while the current cloned policy
supplies every other action.  The resulting mixed-policy transitions are saved
in the historical ``demonstrations.pt`` format consumed by IBMARL, RLfD, and
RFT.
"""

from __future__ import annotations

import argparse
import abc
import copy
import json
import os
import platform
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tensordict import TensorDict
from torch.utils.data import DataLoader
from vmas import make_env

from src.environment.make_env import resolve_scenario
from src.environment.transforms.registry import build_transforms
from src.r2bc.mabc import BCDataset, DecentralizedMiniBC


SUPPORTED_SCENARIOS = ("navigation", "balance", "transport", "buzz_wire")
EXPERIMENT_ALIASES = {"r2bc": "r2bc_decent", "r2bc_decent": "r2bc_decent"}


def _configure_hetgppo_ray_scratch_dir(path_utils: Any) -> Path:
    """Redirect HetGPPO's machine-specific Ray directory without editing it.

    HetGPPO hard-codes ``/local/scratch/mb2389`` on Linux and passes that path
    explicitly to ``ray.init``. The normal Ray temporary-directory variables
    therefore cannot take effect. Patch its small path holder after importing
    the dependency, but before it initializes Ray. The override is process
    local and leaves the pinned checkout unchanged.
    """
    configured_path = os.environ.get("IBMARL_RAY_SCRATCH_DIR")
    scratch_dir = (
        Path(configured_path).expanduser()
        if configured_path
        else Path(tempfile.gettempdir()) / "ibmarl-ray"
    ).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    path_utils.scratch_dir = scratch_dir
    return scratch_dir


def _configure_hetgppo_torch_geometric_compatibility() -> None:
    """Make the pinned HetGPPO ``RelVel`` transform work with recent PyG.

    Older PyG releases allowed transforms to implement ``__call__`` directly.
    Current releases require ``BaseTransform.forward``. HetGPPO predates that
    change, so add the equivalent method to its already-imported class at
    runtime. This is process-local and keeps the external checkout pristine.
    """
    from models.gppo import RelVel

    if "forward" in getattr(RelVel, "__abstractmethods__", frozenset()):
        RelVel.forward = RelVel.__call__
        abc.update_abstractmethods(RelVel)
        print("Applied HetGPPO compatibility for current torch_geometric")


def _configure_legacy_r2bc_checkpoint_imports(multi_trainer_module: Any) -> None:
    """Expose the package name recorded in legacy R2BC Ray checkpoints.

    The supplied MAPPO supervisors were saved when the communication library
    lived at ``src.external_libs``. It is now kept under ``src/r2bc`` and
    imported as a top-level package for compatibility with HetGPPO. Register
    aliases only for pickle deserialization; no dependency files are changed.
    """
    import src as ibmarl_src

    legacy_external_name = "src.external_libs"
    legacy_package = sys.modules.get(legacy_external_name)
    if legacy_package is None:
        legacy_package = types.ModuleType(legacy_external_name)
        legacy_package.__path__ = []
        sys.modules[legacy_external_name] = legacy_package
        setattr(ibmarl_src, "external_libs", legacy_package)

    legacy_comms_name = f"{legacy_external_name}.rllib_differentiable_comms"
    modern_comms_name = "rllib_differentiable_comms"
    modern_comms = sys.modules[modern_comms_name]
    sys.modules[legacy_comms_name] = modern_comms
    sys.modules[f"{legacy_comms_name}.multi_trainer"] = multi_trainer_module
    setattr(legacy_package, "rllib_differentiable_comms", modern_comms)


class NativeRewardTransformedVmasEnv:
    """Apply IBMARL's TorchRL reward transforms to R2BC's native VMAS API.

    R2BC deliberately keeps its original ``list[Tensor]`` VMAS interaction
    interface, while IBMARL's transforms operate on TensorDicts. This adapter
    only converts the just-stepped observations and rewards at that boundary;
    reward semantics remain defined exclusively in ``src/environment/transforms``.
    """

    def __init__(self, env, transforms):
        self._env = env
        self._transforms = list(transforms)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, actions):
        result = list(self._env.step(actions))
        observations, rewards, dones = result[0], result[1], result[2]
        observation_tensor = torch.stack(observations, dim=1)
        # Preserve VMAS's native reward rank while making the agent axis
        # explicit (current VMAS balance rewards are ``[batch]`` per agent).
        reward_tensor = torch.stack(rewards, dim=1)
        td = TensorDict(
            {
                ("agents", "observation"): observation_tensor,
                ("agents", "reward"): reward_tensor,
                "done": torch.as_tensor(dones, dtype=torch.bool),
                "terminated": torch.as_tensor(dones, dtype=torch.bool),
            },
            batch_size=[observation_tensor.shape[0]],
        )
        for transform in self._transforms:
            td = transform._call(td)
        transformed_rewards = td.get(("agents", "reward"))
        result[1] = [
            transformed_rewards[:, agent_id].to(rewards[agent_id].dtype)
            for agent_id in range(transformed_rewards.shape[1])
        ]
        result[2] = td.get("done")
        return tuple(result)


@dataclass
class TrainingConfig:
    total_demonstrations: int = 500
    total_samples: int = 10
    batch_size: int = 256
    shuffle_data: bool = True
    alpha: float = 0.9


def _merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Apply the R2BC environment overlay, merging ``training`` key by key."""
    merged = copy.deepcopy(base)
    overlay = copy.deepcopy(overlay or {})
    if "training" in overlay:
        merged.setdefault("training", {}).update(overlay.pop("training") or {})
    merged.update(overlay)
    return merged


def load_config(scenario_name: str) -> dict[str, Any]:
    if scenario_name not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported R2BC scenario '{scenario_name}'. "
            f"Expected one of: {', '.join(SUPPORTED_SCENARIOS)}"
        )
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "config" / "r2bc"
    with (config_dir / "base.yaml").open() as f:
        config = yaml.safe_load(f) or {}
    with (config_dir / "environments" / f"{scenario_name}.yaml").open() as f:
        config = _merge_config(config, yaml.safe_load(f) or {})
    config["scenario_name"] = scenario_name
    config["exp_type"] = "r2bc_decent"
    return config


class _BuzzWireHeuristic:
    """Exact R2BC SimpleBuzzwireHeuristic, kept local to this runner."""

    def __init__(self, target_x_abs: float = 0.6, move_factor: float = 0.5):
        self.target_x_abs = target_x_abs
        self.move_factor = move_factor

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        agent_x, agent_y = observation[:, 0], observation[:, 1]
        rel_y = observation[:, 5]
        desired_x = torch.sign(agent_x + 1e-9) * self.target_x_abs
        goal_y = agent_y - rel_y
        return torch.stack(
            (
                self.move_factor * (desired_x - agent_x),
                self.move_factor * (goal_y - agent_y),
            ),
            dim=-1,
        )


def _build_heuristic(scenario_name: str):
    if scenario_name == "balance":
        from vmas.scenarios.balance import HeuristicPolicy
        return HeuristicPolicy(continuous_action=True)
    if scenario_name == "navigation":
        from vmas.scenarios.navigation import HeuristicPolicy
        return HeuristicPolicy(continuous_action=True)
    if scenario_name == "transport":
        from vmas.scenarios.transport import HeuristicPolicy
        return HeuristicPolicy(continuous_action=True)
    if scenario_name == "buzz_wire":
        return _BuzzWireHeuristic()
    raise ValueError(f"No R2BC heuristic exists for {scenario_name}")


class LearnedRllibPolicy:
    """The R2BC Ray/RLlib learned-supervisor wrapper.

    Imports remain lazy so people collecting with a VMAS heuristic do not need
    Ray. MAPPO/IPPO use the exact two external R2BC source dependencies; the
    error tells users how to initialize them instead of failing at module import
    time.
    """

    def __init__(self, scenario_name: str, policy_type: str, checkpoint: str, n_agents: int):
        try:
            import ray
            from ray.tune import register_env
            from ray.rllib.agents.ppo import PPOTrainer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "Learned R2BC supervisors require the optional R2BC stack. "
                "Install `requirements-r2bc.txt`; for MAPPO/IPPO also follow "
                "`src/r2bc/external_libs/README.md` to clone the pinned "
                "source dependencies."
            ) from exc

        base_env_config = {
            "device": "cpu",
            "num_envs": 1,
            "scenario_name": scenario_name,
            "continuous_actions": True,
            "max_steps": 500,
            "scenario_config": {"n_agents": n_agents},
        }
        config = {
            "framework": "torch", "num_gpus": 0, "num_workers": 0,
            "train_batch_size": 60000, "rollout_fragment_length": 125,
            "sgd_minibatch_size": 4096, "explore": False,
            "env_config": base_env_config, "env": scenario_name,
        }

        def env_creator(env_config):
            return make_env(
                scenario=scenario_name,
                num_envs=1,
                wrapper=None,
                **env_config,
            )

        register_env(scenario_name, lambda _cfg: env_creator(base_env_config))
        if policy_type in ("learned", "CPPO"):
            trainer = PPOTrainer(config=config, env=scenario_name)
        else:  # MAPPO/IPPO retain the original R2BC custom trainer path.
            try:
                external_libs = Path(__file__).resolve().parent / "external_libs"
                hetgppo = external_libs / "HetGPPO"
                differentiable_comms = external_libs / "rllib_differentiable_comms"
                if not (hetgppo / "utils.py").is_file() or not differentiable_comms.is_dir():
                    raise ModuleNotFoundError(
                        "Pinned R2BC source directories are missing from "
                        f"{external_libs}"
                    )

                # HetGPPO's pinned source uses top-level imports such as
                # ``from models...`` and ``from evaluate...``. R2BC originally
                # executed with its checkout on sys.path, so reproduce that
                # import context instead of treating it as a normal package.
                for source_dir in (str(external_libs), str(hetgppo)):
                    if source_dir not in sys.path:
                        sys.path.insert(0, source_dir)
                from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
                from utils import PathUtils, TrainingUtils
            except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional stack
                raise RuntimeError(
                    f"Could not import the R2BC {policy_type} supervisor stack: "
                    f"{type(exc).__name__}: {exc}. Reinstall "
                    "`requirements-r2bc.txt` and verify the pinned checkouts "
                    "in `src/r2bc/external_libs/README.md`."
            ) from exc
            if not ray.is_initialized():
                ray_scratch_dir = _configure_hetgppo_ray_scratch_dir(PathUtils)
                print(f"Using Ray scratch directory: {ray_scratch_dir}")
                TrainingUtils.init_ray(scenario_name=scenario_name, local_mode=False)
            _configure_hetgppo_torch_geometric_compatibility()
            _configure_legacy_r2bc_checkpoint_imports(
                sys.modules[MultiPPOTrainer.__module__]
            )
            algo = {
                "CPPO": (True, False), "MAPPO": (False, True),
                "IPPO": (False, False),
            }[policy_type]
            config["model"] = {
                "custom_model": "GPPO", "custom_action_dist": "hom_multi_action",
                "custom_model_config": {
                    "activation_fn": "relu", "heterogeneous": False,
                    "comm_radius": 0.35, "pos_start": 0, "pos_dim": 2,
                    "vel_start": 2, "vel_dim": 2, "trainer": "MultiPPOTrainer",
                    "share_action_value": False, "use_beta": False,
                    "add_agent_index": False, "aggr": "add", "gnn_type": "MatPosConv",
                    "share_observations": algo[0], "centralised_critic": algo[1],
                    "use_mlp": False,
                },
            }
            trainer = MultiPPOTrainer(config=config, env=scenario_name)
        trainer.restore(checkpoint)
        self.policy = trainer.get_policy()

    def compute_action(self, observation: torch.Tensor) -> torch.Tensor:
        actions, _, _ = self.policy.compute_actions(observation)
        return torch.tensor([action.tolist() for action in actions], dtype=torch.float32)


class R2BCExperiment:
    """The R2BC decentralized trainer and IBMARL artifact writer."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scenario_name = config["scenario_name"]
        self.n_agents = int(config["n"])
        self.horizon = int(config["horizon"])
        self.seed = config.get("seed")
        self.should_render = bool(config.get("should_render", False))
        self.sparse_rewards = bool(config.get("sparse_rewards", False))
        self._environment_mode_logged = False
        self.training = TrainingConfig(**config["training"])
        if self.training.total_samples <= 0:
            raise ValueError("training.total_samples must be positive")
        self.demos_per_eval = self.training.total_demonstrations // self.training.total_samples
        if self.demos_per_eval < self.n_agents:
            raise ValueError(
                "total_demonstrations // total_samples must be at least the number of agents "
                "so each R2BC evaluation collects one demonstration per agent."
            )
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        probe = self._make_env(num_envs=1, seed=self.seed)
        probe_obs = probe.reset(seed=self.seed)
        self.in_dims = sum(obs.shape[-1] for obs in probe_obs)
        self.out_dims = self.n_agents * 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = DecentralizedMiniBC(
            self.n_agents, self.in_dims, self.out_dims,
            hidden_size=int(config["hidden_size"]),
            hidden_layers=int(config["hidden_layers"]),
        ).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()
        self.expert_pth = config.get("expert_pth")
        if not self.expert_pth:
            raise ValueError("--expert_pth is required (or pass --expert_pth heuristic).")
        self.expert_policy_type = config.get("expert_policy_type", "learned")
        self.expert_policy = (
            _build_heuristic(self.scenario_name)
            if self.expert_pth == "heuristic"
            else LearnedRllibPolicy(
                self.scenario_name, self.expert_policy_type, self.expert_pth, self.n_agents
            )
        )
        self.online_data = {"obs": [], "act": [], "indices": []}
        self.demo_buffer = {"obs": [], "act": [], "rewards": [], "next_obs": [], "dones": []}
        self.demos_collected_per_agent = [0] * self.n_agents
        self.results = {
            "total_demonstrations": [], "average_rewards": [],
            "on_robot_errors": [], "expert_distribution_errors": [],
        }

    def _make_env(self, num_envs: int, seed: int | None):
        scenario = resolve_scenario(self.config)
        kwargs: dict[str, Any] = {
            "scenario": (
                scenario
            ), "num_envs": num_envs, "device": "cpu",
            "continuous_actions": True, "max_steps": self.horizon, "seed": seed,
        }
        if self.scenario_name in ("navigation", "transport"):
            kwargs["n_agents"] = self.n_agents
        if self.scenario_name == "transport" and "n_packages" in self.config:
            kwargs["n_packages"] = self.config["n_packages"]
        env = make_env(**kwargs)
        transforms = build_transforms(self.config)
        if transforms:
            if not self._environment_mode_logged:
                print(
                    "R2BC using IBMARL sparse-reward transforms:",
                    [type(item).__name__ for item in transforms],
                )
                self._environment_mode_logged = True
            return NativeRewardTransformedVmasEnv(env, transforms)
        if self.sparse_rewards and self.scenario_name == "buzz_wire":
            if not self._environment_mode_logged:
                print("R2BC using IBMARL's binary sparse buzz_wire reward")
                self._environment_mode_logged = True
            return env
        if self.sparse_rewards and self.scenario_name == "balance":
            if not self._environment_mode_logged:
                print("R2BC using IBMARL's native-predicate sparse balance reward")
                self._environment_mode_logged = True
            return env
        if not self._environment_mode_logged:
            print("R2BC using native VMAS dense rewards (sparse_rewards: false)")
            self._environment_mode_logged = True
        return env

    def _expert_action(self, flat_obs: torch.Tensor) -> torch.Tensor:
        if self.expert_pth == "heuristic":
            local_obs = flat_obs.reshape(-1, self.in_dims // self.n_agents)
            action = self.expert_policy.compute_action(local_obs, 1.0)
        else:
            action = self.expert_policy.compute_action(flat_obs)
            # Preserve R2BC's learned-policy layout conversion exactly.
            action = torch.swapaxes(action, 0, 1)
        # RLlib's Gaussian policy can produce a value just outside VMAS's
        # strict physical-action range. Clamp at the environment boundary so
        # one supervisor sample cannot abort a collection rollout.
        return torch.clamp(
            torch.as_tensor(action, dtype=torch.float32), -1.0, 1.0
        ).reshape(-1, self.n_agents, 1, 2)

    def _compute_loss(self, predicted: torch.Tensor, target: torch.Tensor, index: torch.Tensor):
        if target.ndim == 3:
            target = target.squeeze()
        rows = torch.arange(predicted.shape[0], device=predicted.device)
        action_0 = predicted[rows, index * 2]
        action_1 = predicted[rows, index * 2 + 1]
        return self.loss_fn(torch.stack((action_0, action_1), dim=1), target.float())

    def _train_one_iter(self, dataloader: DataLoader) -> list[float]:
        losses = []
        for states, actions, indices in dataloader:
            states, actions, indices = states.to(self.device), actions.to(self.device), indices.to(self.device)
            self.optim.zero_grad()
            predicted = torch.clamp(self.policy(states.float()), -1, 1)
            loss = self._compute_loss(predicted, actions, indices)
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        return losses

    def new_demonstrations(self, n_evals: int, agent_id: int):
        if self.seed is None:
            demo_seed = None
        else:
            demo_seed = self.seed * 1_000_000 + agent_id * 100_000 + self.demos_collected_per_agent[agent_id]
        env = self._make_env(n_evals, demo_seed)
        self.demos_collected_per_agent[agent_id] += n_evals
        obs = env.reset(seed=None)
        big_obs, big_actions = None, None
        indices = []
        active = torch.ones(n_evals, dtype=torch.bool)

        for _ in range(self.horizon):
            flat_obs = torch.cat(obs, dim=1)
            with torch.no_grad():
                policy_actions = torch.clamp(self.policy(flat_obs.to(self.device).float()), -1, 1).cpu()
            policy_actions = policy_actions.reshape(n_evals, self.n_agents, 1, 2)
            expert_actions = self._expert_action(flat_obs)
            blended_actions = policy_actions.clone()
            blended_actions[:, agent_id] = expert_actions[:, agent_id]

            active_obs = flat_obs[active]
            big_obs = torch.cat((big_obs, active_obs), dim=0) if big_obs is not None else active_obs
            agent_actions = blended_actions[active, agent_id]
            big_actions = torch.cat((big_actions, agent_actions), dim=0) if big_actions is not None else agent_actions
            indices.extend([agent_id] * int(active.sum().item()))

            env_actions = torch.swapaxes(blended_actions, 0, 1).squeeze(2)
            previous_obs = obs
            obs, rewards, dones, _ = env.step(env_actions)
            self.demo_buffer["obs"].extend(torch.stack(previous_obs, dim=1)[active].cpu().numpy())
            self.demo_buffer["act"].extend(env_actions.permute(1, 0, 2)[active].cpu().numpy())
            self.demo_buffer["rewards"].extend(torch.stack(rewards, dim=1)[active].cpu().numpy())
            self.demo_buffer["next_obs"].extend(torch.stack(obs, dim=1)[active].cpu().numpy())
            done_tensor = torch.as_tensor(dones, dtype=torch.bool).reshape(-1)
            self.demo_buffer["dones"].extend(done_tensor[active].cpu().numpy())
            active &= ~done_tensor
            if self.should_render:
                env.render()
            if not active.any():
                break

        return big_obs.tolist(), big_actions.reshape(-1, 2).tolist(), indices

    def test_batched_rollouts(self, n_evals: int = 50, save_gif_path: Path | None = None):
        env = self._make_env(n_evals, seed=None)
        obs = env.reset(seed=0)
        episode_rewards = torch.zeros(n_evals)
        active = torch.ones(n_evals, dtype=torch.bool)
        cumulative_error = 0.0
        frames = []
        for _ in range(self.horizon):
            flat_obs = torch.cat(obs, dim=1)
            with torch.no_grad():
                actions = torch.clamp(self.policy(flat_obs.to(self.device).float()), -1, 1).cpu()
            action_batch = actions.reshape(n_evals, self.n_agents, 1, 2)
            obs, reward, dones, _ = env.step(torch.swapaxes(action_batch, 0, 1).squeeze(2))
            if save_gif_path is not None:
                frame = env.render(mode="rgb_array", agent_index_focus=None, env_index=0)
                if frame is not None:
                    frames.append(frame)
            episode_rewards += reward[0] * active.to(reward[0].dtype)
            expert_actions = self._expert_action(flat_obs)
            sampled = torch.randint(0, self.n_agents, (n_evals,))
            cumulative_error += self.loss_fn(action_batch[:, sampled], expert_actions[:, sampled]).item()
            active &= ~torch.as_tensor(dones, dtype=torch.bool).reshape(-1)
            if not active.any():
                break
        if save_gif_path is not None and frames:
            imageio.mimsave(save_gif_path, frames, duration=100)
        return episode_rewards.mean().item(), cumulative_error / n_evals

    def train(self):
        demos_collected = 0
        demos_needed = self.demos_per_eval // self.n_agents
        while demos_collected < self.training.total_demonstrations:
            for agent_id in range(self.n_agents):
                observations, actions, indices = self.new_demonstrations(demos_needed, agent_id)
                self.online_data["obs"].extend(observations)
                self.online_data["act"].extend(actions)
                self.online_data["indices"].extend(indices)
                demos_collected += demos_needed

            dataset = BCDataset(
                np.array(self.online_data["obs"]), np.array(self.online_data["act"]),
                np.array(self.online_data["indices"]),
            )
            dataloader = DataLoader(dataset, batch_size=self.training.batch_size, shuffle=self.training.shuffle_data)
            previous_loss, current_loss = 1e9, 1e8
            while previous_loss - current_loss > 1e-4:
                previous_loss = current_loss
                losses = self._train_one_iter(dataloader)
                current_loss = sum(losses) / len(losses)
            avg_reward, deployment_error = self.test_batched_rollouts()
            self.results["total_demonstrations"].append(demos_collected)
            self.results["average_rewards"].append(avg_reward)
            self.results["on_robot_errors"].append(deployment_error)
            self.results["expert_distribution_errors"].append(current_loss)
            print(f"Evaluated at {demos_collected} demonstrations: reward={avg_reward:.3f}")
        return self.save_exp_data()

    def save_exp_data(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        directory_name = f"{self.scenario_name.replace('buzz_wire', 'buzzwire')}_r2bc_decent_{timestamp}"
        exp_dir = Path("results") / directory_name
        exp_dir.mkdir(parents=True, exist_ok=False)
        self.policy.save_checkpoint(exp_dir / "policy_checkpoint.pth")
        torch.save(self.demo_buffer, exp_dir / "demonstrations.pt")
        try:
            self.test_batched_rollouts(n_evals=1, save_gif_path=exp_dir / "performance_demo.gif")
        except Exception as exc:
            print(f"Warning: could not generate R2BC demonstration gif: {exc}")
        pd.DataFrame(self.results).to_csv(exp_dir / "metrics.csv", index=False)
        with (exp_dir / "config.yaml").open("w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        with (exp_dir / "metadata.json").open("w") as f:
            json.dump({
                "timestamp": timestamp, "python_version": platform.python_version(),
                "command_line": " ".join(sys.argv), "working_directory": str(Path.cwd()),
            }, f, indent=2)
        fig, (reward_axis, error_axis) = plt.subplots(2, 1, figsize=(10, 12))
        reward_axis.plot(self.results["total_demonstrations"], self.results["average_rewards"], marker="o")
        reward_axis.set(xlabel="Demonstrations", ylabel="Average reward")
        error_axis.plot(self.results["total_demonstrations"], self.results["expert_distribution_errors"], label="In-distribution")
        error_axis.plot(self.results["total_demonstrations"], self.results["on_robot_errors"], label="Deployment")
        error_axis.set(xlabel="Demonstrations", ylabel="MSE")
        error_axis.legend()
        fig.tight_layout()
        fig.savefig(exp_dir / "training_metrics.png", dpi=300)
        plt.close(fig)
        print(f"R2BC artifacts saved to: {exp_dir.resolve()}")
        return exp_dir


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train an IBMARL R2BC demonstrator")
    parser.add_argument("scenario_name", choices=SUPPORTED_SCENARIOS)
    parser.add_argument("exp_type", choices=tuple(EXPERIMENT_ALIASES), help="Use 'r2bc' for decentralized R2BC")
    parser.add_argument(
        "--expert_pth",
        default=None,
        help="Override the scenario-configured RLlib checkpoint path, or use 'heuristic'",
    )
    parser.add_argument("--expert_policy_type", choices=("learned", "CPPO", "MAPPO", "IPPO"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--hidden_layers", type=int, default=None)
    parser.add_argument("--total_demonstrations", "--total-demonstrations", type=int, default=None)
    parser.add_argument("--total_samples", "--total-samples", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    reward_mode = parser.add_mutually_exclusive_group()
    reward_mode.add_argument("--sparse", dest="sparse_rewards", action="store_true", default=None)
    reward_mode.add_argument("--dense", dest="sparse_rewards", action="store_false", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = load_config(args.scenario_name)
    config["exp_type"] = EXPERIMENT_ALIASES[args.exp_type]
    if args.expert_pth is not None:
        config["expert_pth"] = args.expert_pth
    if args.expert_policy_type is not None:
        config["expert_policy_type"] = args.expert_policy_type
    for key in ("seed", "horizon", "n", "hidden_size", "hidden_layers"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("total_demonstrations", "total_samples"):
        value = getattr(args, key)
        if value is not None:
            config["training"][key] = value
    if args.sparse_rewards is not None:
        config["sparse_rewards"] = bool(args.sparse_rewards)
    config["should_render"] = bool(args.render)
    return R2BCExperiment(config).train()


if __name__ == "__main__":
    main()
