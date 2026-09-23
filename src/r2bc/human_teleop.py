"""Train an R2BC teacher from round-robin human Xbox demonstrations.

One human controls one agent per episode. All remaining agents execute their
current decentralized BC policies. After each complete round, the policies are
updated from only the actions demonstrated for their respective agents. The
mixed rollouts are also exported in IBMARL's historical ``demonstrations.pt``
transition format.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchrl.envs.utils import step_mdp

from src.environment.make_env import make_env
from src.r2bc.human_gamepad import XboxGamepad
from src.r2bc.mabc import DecentralizedMiniBC
from src.run_experiment import load_config


SUPPORTED_SCENARIOS = ("buzz_wire", "transport")
AGENT_COLORS = (
    ("BLUE", (0.15, 0.35, 0.95)),
    ("ORANGE", (0.95, 0.45, 0.05)),
    ("GREEN", (0.10, 0.75, 0.25)),
)


@dataclass
class DemonstrationBuffer:
    obs: list[torch.Tensor] = field(default_factory=list)
    act: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    next_obs: list[torch.Tensor] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    controlled_agent: list[int] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "DemonstrationBuffer":
        """Rebuild a human demonstration buffer from its on-disk artifact."""
        meta = data.get("meta", {})
        controlled_agent = meta.get("controlled_agent")
        episodes = meta.get("episodes")
        if controlled_agent is None or episodes is None:
            raise ValueError(
                "Cannot resume: demonstrations.pt has no human R2BC "
                "controlled-agent/episode metadata."
            )

        count = len(data["obs"])
        fields = ("act", "rewards", "next_obs", "dones")
        if any(len(data[name]) != count for name in fields) or len(controlled_agent) != count:
            raise ValueError("Cannot resume: demonstrations.pt fields have different lengths.")

        terminated = data.get("terminated", torch.zeros(count, dtype=torch.bool))
        if len(terminated) != count:
            raise ValueError("Cannot resume: demonstrations.pt terminated has the wrong length.")

        buffer = cls()
        buffer.obs = [value.detach().cpu().to(torch.float32) for value in data["obs"]]
        buffer.act = [value.detach().cpu().to(torch.float32) for value in data["act"]]
        buffer.rewards = [value.detach().cpu().to(torch.float32) for value in data["rewards"]]
        buffer.next_obs = [value.detach().cpu().to(torch.float32) for value in data["next_obs"]]
        buffer.dones = [bool(value) for value in data["dones"]]
        buffer.terminated = [bool(value) for value in terminated]
        buffer.controlled_agent = [int(value) for value in controlled_agent]
        buffer.episodes = list(episodes)
        return buffer

    def add_transition(self, obs, act, reward, next_obs, done, terminated, agent_id):
        self.obs.append(obs.detach().cpu().to(torch.float32))
        self.act.append(act.detach().cpu().to(torch.float32))
        self.rewards.append(reward.detach().cpu().to(torch.float32))
        self.next_obs.append(next_obs.detach().cpu().to(torch.float32))
        self.dones.append(bool(done))
        self.terminated.append(bool(terminated))
        self.controlled_agent.append(int(agent_id))

    def as_dict(self, scenario: str, reward_mode: str | None = None) -> dict:
        if not self.obs:
            raise RuntimeError("No accepted human demonstrations have been collected.")
        return {
            "obs": torch.stack(self.obs),
            "act": torch.stack(self.act),
            "rewards": torch.stack(self.rewards),
            "next_obs": torch.stack(self.next_obs),
            "dones": torch.tensor(self.dones, dtype=torch.bool),
            "terminated": torch.tensor(self.terminated, dtype=torch.bool),
            "meta": {
                "format_version": 2,
                "source": "human_xbox_r2bc",
                "scenario": scenario,
                "reward_mode": reward_mode,
                "controlled_agent": torch.tensor(self.controlled_agent, dtype=torch.long),
                "episodes": self.episodes,
            },
        }


class HumanR2BCTrainer:
    def __init__(
        self,
        config: dict,
        args: argparse.Namespace,
        output_dir: Path,
        resume: bool = False,
    ):
        self.config = dict(config)
        self.args = args
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        torch.manual_seed(args.seed)

        # The interactive collector must contain exactly one VMAS sub-environment.
        self.config["frames_per_batch"] = self.config["horizon"]
        self.config["seed"] = args.seed
        self.env = make_env(self.config, self.device)
        self.group = next(iter(self.env.group_map))
        self.n_agents = len(self.env.group_map[self.group])
        self.obs_dim = self.env.observation_spec[self.group, "observation"].shape[-1]
        self.act_dim = self.env.full_action_spec[self.group, "action"].shape[-1]
        self._vmas_env = self.env.base_env._env
        self._agent_overlay = None
        self._apply_agent_colors()
        if resume:
            self.policy = DecentralizedMiniBC.load_checkpoint(
                output_dir / "policy_checkpoint.pth", self.device
            )
            if (
                self.policy.n != self.n_agents
                or self.policy.in_size != self.obs_dim
                or self.policy.out_size != self.act_dim
            ):
                raise ValueError(
                    "Cannot resume: checkpoint dimensions do not match the saved environment."
                )
        else:
            self.policy = DecentralizedMiniBC(
                self.n_agents,
                self.n_agents * self.obs_dim,
                self.n_agents * self.act_dim,
                hidden_size=args.hidden_size,
                hidden_layers=args.hidden_layers,
            ).to(self.device)
        self.optimizers = [
            torch.optim.Adam(getattr(self.policy, f"pi_{i}").parameters(), lr=args.lr)
            for i in range(self.n_agents)
        ]
        self.agent_obs: list[list[torch.Tensor]] = [[] for _ in range(self.n_agents)]
        self.agent_act: list[list[torch.Tensor]] = [[] for _ in range(self.n_agents)]
        self.demo_buffer = DemonstrationBuffer()
        self._prior_metadata = {}
        if resume:
            self._restore_training_data()
        self.gamepad = XboxGamepad(
            index=args.controller,
            speed=args.speed,
            precision_speed=args.precision_speed,
            deadzone=args.deadzone,
        )

    def _restore_training_data(self) -> None:
        demonstrations = torch.load(
            self.output_dir / "demonstrations.pt", map_location="cpu"
        )
        self.demo_buffer = DemonstrationBuffer.from_dict(demonstrations)
        for obs, action, agent_id in zip(
            self.demo_buffer.obs,
            self.demo_buffer.act,
            self.demo_buffer.controlled_agent,
        ):
            if not 0 <= agent_id < self.n_agents:
                raise ValueError(f"Cannot resume: invalid controlled agent {agent_id}.")
            self.agent_obs[agent_id].append(obs[agent_id].clone())
            self.agent_act[agent_id].append(action[agent_id].clone())

        state_path = self.output_dir / "training_state.pth"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            if state.get("episodes") != len(self.demo_buffer.episodes):
                raise ValueError(
                    "Cannot resume: training_state.pth and demonstrations.pt "
                    "describe different save points."
                )
            optimizer_states = state.get("optimizers", [])
            if len(optimizer_states) != len(self.optimizers):
                raise ValueError("Cannot resume: training_state.pth has the wrong optimizer count.")
            for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
                optimizer.load_state_dict(optimizer_state)
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
        else:
            print("No training_state.pth found; continuing with fresh Adam optimizer state.")

        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open() as f:
                self._prior_metadata = json.load(f)
        self._prior_metadata["resume_count"] = self._prior_metadata.get("resume_count", 0) + 1
        print(
            f"Resumed {len(self.demo_buffer.episodes)} accepted episodes and "
            f"{len(self.demo_buffer.obs)} transitions from {self.output_dir.resolve()}"
        )

    def _apply_agent_colors(self) -> None:
        """Give agents stable high-contrast identities in every rendered frame."""
        for index, agent in enumerate(self._vmas_env.agents):
            agent.color = AGENT_COLORS[index][1]
            agent._alpha = 0.9

    def _update_agent_overlay(self, agent_id: int) -> None:
        """Create/update the persistent in-window agent legend and selection."""
        if self._agent_overlay is None:
            from vmas.simulator import rendering

            legend = rendering.TextLine(x=10, y=10, font_size=13)
            selected = rendering.TextLine(x=10, y=32, font_size=15)
            self._vmas_env.viewer.add_geom(legend)
            self._vmas_env.viewer.add_geom(selected)
            self._agent_overlay = (legend, selected)

        legend_text = "   |   ".join(
            f"Agent {index + 1} = {AGENT_COLORS[index][0]}"
            for index in range(self.n_agents)
        )
        selected_color = AGENT_COLORS[agent_id][0]
        self._agent_overlay[0].set_text(legend_text)
        self._agent_overlay[1].set_text(
            f"YOU CONTROL: Agent {agent_id + 1} ({selected_color})"
        )

    @torch.no_grad()
    def _policy_actions(self, obs: torch.Tensor) -> torch.Tensor:
        flat = obs.reshape(obs.shape[0], -1)
        return torch.clamp(self.policy(flat), -1.0, 1.0).reshape(
            obs.shape[0], self.n_agents, self.act_dim
        )

    def _wait_for_start(self, agent_id: int) -> bool:
        selected_color = AGENT_COLORS[agent_id][0]
        print(
            f"\nAgent {agent_id + 1}/{self.n_agents} ({selected_color}): "
            "A=start, B=reject during rollout, "
            "START=save and quit. RT enables precision mode."
        )
        while True:
            self.env.render()
            state = self.gamepad.poll()
            if state.quit:
                return False
            if state.start:
                return True
            time.sleep(0.01)

    def _collect_episode(self, agent_id: int, episode_index: int) -> bool:
        td = self.env.reset()
        self.env.render()
        self._update_agent_overlay(agent_id)
        if not self._wait_for_start(agent_id):
            raise KeyboardInterrupt
        staged = []
        total_reward = 0.0

        for step in range(self.config["horizon"]):
            state = self.gamepad.poll()
            if state.quit:
                raise KeyboardInterrupt
            if state.reject:
                print("Episode rejected; resetting without adding it to the dataset.")
                return False

            obs = td[self.group, "observation"]
            action = self._policy_actions(obs)
            human_action = torch.as_tensor(state.action, dtype=action.dtype)
            action[0, agent_id, : min(self.act_dim, 2)] = human_action[: self.act_dim]
            td[self.group, "action"] = action
            transition = self.env.step(td)
            next_td = transition["next"]
            reward = next_td[self.group, "reward"]
            done = bool(next_td["done"].reshape(-1)[0].item())
            terminated_td = next_td.get("terminated")
            terminated = bool(terminated_td.reshape(-1)[0].item()) if terminated_td is not None else False
            staged.append((
                obs[0].clone(),
                action[0].clone(),
                reward[0].clone(),
                next_td[self.group, "observation"][0].clone(),
                done,
                terminated,
            ))
            total_reward += float(reward[0].mean().item())
            self.env.render()
            if done:
                break
            td = step_mdp(transition, keep_other=True)
            time.sleep(self.args.step_delay)

        for obs, action, reward, next_obs, done, terminated in staged:
            self.demo_buffer.add_transition(obs, action, reward, next_obs, done, terminated, agent_id)
            self.agent_obs[agent_id].append(obs[agent_id].clone())
            self.agent_act[agent_id].append(action[agent_id].clone())
        self.demo_buffer.episodes.append({
            "episode": episode_index,
            "controlled_agent": agent_id,
            "length": len(staged),
            "return_mean": total_reward,
        })
        print(f"Accepted episode {episode_index + 1}: {len(staged)} steps, return {total_reward:.3f}")
        return True

    def _train_policies(self) -> None:
        self.policy.train()
        for agent_id in range(self.n_agents):
            if not self.agent_obs[agent_id]:
                continue
            dataset = TensorDataset(
                torch.stack(self.agent_obs[agent_id]),
                torch.stack(self.agent_act[agent_id]),
            )
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
            model = getattr(self.policy, f"pi_{agent_id}")
            optimizer = self.optimizers[agent_id]
            final_loss = 0.0
            for _ in range(self.args.epochs):
                weighted_loss = 0.0
                seen = 0
                for obs, action in loader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(torch.clamp(model(obs), -1, 1), action)
                    loss.backward()
                    optimizer.step()
                    weighted_loss += loss.item() * len(obs)
                    seen += len(obs)
                final_loss = weighted_loss / max(seen, 1)
            print(f"Trained agent {agent_id + 1} on {len(dataset)} labels; MSE={final_loss:.6f}")
        self.policy.eval()

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.output_dir / "policy_checkpoint.pth"
        checkpoint_tmp = checkpoint_path.with_suffix(".pth.tmp")
        self.policy.save_checkpoint(checkpoint_tmp)
        checkpoint_tmp.replace(checkpoint_path)
        demonstrations_path = self.output_dir / "demonstrations.pt"
        demonstrations_tmp = demonstrations_path.with_suffix(".pt.tmp")
        torch.save(
            self.demo_buffer.as_dict(
                self.config["scenario_name"],
                "sparse" if self.config.get("sparse_rewards", False) else "dense",
            ),
            demonstrations_tmp,
        )
        demonstrations_tmp.replace(demonstrations_path)
        state_path = self.output_dir / "training_state.pth"
        state_tmp = state_path.with_suffix(".pth.tmp")
        torch.save({
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "torch_rng_state": torch.get_rng_state(),
            "episodes": len(self.demo_buffer.episodes),
        }, state_tmp)
        state_tmp.replace(state_path)
        metadata = dict(self._prior_metadata)
        metadata.setdefault("command_line", " ".join(sys.argv))
        metadata.update({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "python_version": platform.python_version(),
            "last_command_line": " ".join(sys.argv),
            "scenario": self.config["scenario_name"],
            "episodes": len(self.demo_buffer.episodes),
            "transitions": len(self.demo_buffer.obs),
            "artifact_schema": "IBMARL demonstrations.pt",
            "reward_mode": (
                "sparse" if self.config.get("sparse_rewards", False) else "dense"
            ),
        })
        with (self.output_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        self._prior_metadata = metadata
        print(f"Saved R2BC checkpoint and demonstrations.pt to {self.output_dir.resolve()}")

    def train(self) -> None:
        accepted = len(self.demo_buffer.episodes)
        if accepted >= self.args.total_demonstrations:
            print(
                f"Run already has {accepted} demonstrations; target is "
                f"{self.args.total_demonstrations}. Nothing to collect."
            )
        try:
            while accepted < self.args.total_demonstrations:
                agent_id = accepted % self.n_agents
                if not self._collect_episode(agent_id, accepted):
                    continue
                accepted += 1
                if accepted % self.n_agents == 0:
                    self._train_policies()
                self.save()  # Human collection is expensive: autosave every accepted run.
        except KeyboardInterrupt:
            print("Collection stopped by operator.")
        finally:
            if accepted and accepted % self.n_agents:
                self._train_policies()
                self.save()
            self.gamepad.close()
            self.env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a decentralized R2BC demonstrator with one human Xbox controller."
    )
    parser.add_argument("scenario", choices=SUPPORTED_SCENARIOS)
    parser.add_argument("--total-demonstrations", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help=(
            "Continue an existing human R2BC run in place. "
            "--total-demonstrations is the new cumulative target."
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--controller", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--precision-speed", type=float)
    parser.add_argument("--deadzone", type=float)
    parser.add_argument("--step-delay", type=float)
    return parser.parse_args(argv)


def load_human_config(scenario: str) -> dict:
    # In particular, buzz_wire retains config/environments/buzz_wire.yaml's
    # sparse + binary-terminal settings, which make_env resolves to the local
    # SparseRewardBuzzWireScenario rather than native VMAS buzz_wire.
    return load_config(scenario, "r2bc_human")


def main(argv=None):
    args = parse_args(argv)
    if args.resume is not None and args.output_dir is not None:
        raise ValueError("--resume and --output-dir cannot be used together.")

    if args.resume is not None:
        output_dir = args.resume.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {output_dir}")
        config_path = output_dir / "config.yaml"
        with config_path.open() as f:
            saved_config = yaml.safe_load(f)
        saved_scenario = saved_config.get("scenario_name")
        if saved_scenario != args.scenario:
            raise ValueError(
                f"Cannot resume {saved_scenario!r} run as {args.scenario!r}."
            )
        saved_human_args = saved_config.get("human_r2bc", {})
        config = {key: value for key, value in saved_config.items() if key != "human_r2bc"}
    else:
        saved_human_args = {}
        config = load_human_config(args.scenario)
    defaults = {
        "buzz_wire": {"hidden_size": 8, "speed": 0.5, "precision_speed": 0.2},
        "transport": {"hidden_size": 32, "speed": 1.0, "precision_speed": 0.25},
    }[args.scenario]
    defaults.update({
        "seed": 0,
        "controller": 0,
        "hidden_layers": 1,
        "epochs": 50,
        "batch_size": 256,
        "lr": 1e-3,
        "deadzone": 0.1,
        "step_delay": 0.01,
    })
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, saved_human_args.get(key, value))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume is None:
        output_dir = args.output_dir or Path("results") / f"{args.scenario}_r2bc_human_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if args.resume is None:
        with (output_dir / "config.yaml").open("w") as f:
            yaml.safe_dump({**config, "human_r2bc": serializable_args}, f, sort_keys=False)
    HumanR2BCTrainer(config, args, output_dir, resume=args.resume is not None).train()


if __name__ == "__main__":
    main()
