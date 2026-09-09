"""Focused compatibility tests for the in-repository R2BC runner."""

import os
import sys
import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.environment.transforms.registry import build_transforms
from src.environment.scenarios.balance_sparse import SparseRewardBalanceScenario
from src.environment.scenarios.buzz_wire_sparse import SparseRewardBuzzWireScenario
from src.r2bc.run_experiment import (
    EXPERIMENT_ALIASES,
    R2BCExperiment,
    _configure_hetgppo_ray_scratch_dir,
    _configure_hetgppo_torch_geometric_compatibility,
    _configure_legacy_r2bc_checkpoint_imports,
    load_config,
)


class TestR2BCRunner(unittest.TestCase):
    def test_legacy_checkpoint_import_aliases_current_communication_module(self):
        modern_package = type(sys)("rllib_differentiable_comms")
        modern_trainer = type(sys)("rllib_differentiable_comms.multi_trainer")
        with patch.dict(
            sys.modules,
            {
                "rllib_differentiable_comms": modern_package,
                "rllib_differentiable_comms.multi_trainer": modern_trainer,
            },
        ):
            _configure_legacy_r2bc_checkpoint_imports(modern_trainer)
            self.assertIs(
                sys.modules["src.external_libs.rllib_differentiable_comms.multi_trainer"],
                modern_trainer,
            )

    def test_hetgppo_pyg_compatibility_implements_forward_on_legacy_transform(self):
        class BaseTransform(ABC):
            @abstractmethod
            def forward(self, data):
                raise NotImplementedError

        class RelVel(BaseTransform):
            def __call__(self, data):
                return data

        models_module = type(sys)("models")
        gppo_module = type(sys)("models.gppo")
        gppo_module.RelVel = RelVel
        with patch.dict(sys.modules, {"models": models_module, "models.gppo": gppo_module}):
            _configure_hetgppo_torch_geometric_compatibility()

        self.assertEqual(RelVel().forward("value"), "value")

    def test_hetgppo_ray_directory_is_overridden_without_editing_dependency(self):
        class PathUtils:
            scratch_dir = Path("/local/scratch/mb2389")

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "ray-scratch"
            with patch.dict(os.environ, {"IBMARL_RAY_SCRATCH_DIR": str(target)}):
                result = _configure_hetgppo_ray_scratch_dir(PathUtils)

            self.assertEqual(result, target)
            self.assertEqual(PathUtils.scratch_dir, target)
            self.assertTrue(target.exists())

    def test_configs_cover_current_ibmarl_scenarios(self):
        for scenario, expected_agents in {
            "navigation": 3, "balance": 3, "transport": 3, "buzz_wire": 2,
        }.items():
            config = load_config(scenario)
            self.assertEqual(config["n"], expected_agents)
            self.assertEqual(config["exp_type"], "r2bc_decent")
            self.assertTrue(config["sparse_rewards"])
            self.assertIn("expert_pth", config)
        self.assertEqual(EXPERIMENT_ALIASES["r2bc"], "r2bc_decent")
        self.assertEqual(load_config("transport")["expert_pth"], "heuristic")

    def test_sparse_override_uses_ibmarl_transform_registry(self):
        for scenario, transform_name in {
            "navigation": "NavigationSparseReward",
            "transport": "TransportSparseReward",
        }.items():
            config = load_config(scenario)
            config["sparse_rewards"] = True
            self.assertEqual(type(build_transforms(config)[0]).__name__, transform_name)
        self.assertEqual(build_transforms(load_config("balance")), [])

    def test_mixed_rollout_exports_ibmarl_demo_schema(self):
        config = load_config("balance")
        config.update({"expert_pth": "heuristic", "horizon": 3, "seed": 0})
        config["training"] = {
            **config["training"], "total_demonstrations": 3,
            "total_samples": 1, "batch_size": 32,
        }
        experiment = R2BCExperiment(config)
        observations, actions, indices = experiment.new_demonstrations(1, agent_id=0)
        self.assertEqual(len(observations), 3)
        self.assertEqual(len(actions), 3)
        self.assertEqual(len(indices), 3)
        expected_shapes = {
            "obs": (3, 3, 16), "act": (3, 3, 2), "rewards": (3, 3),
            "next_obs": (3, 3, 16), "dones": (3,),
        }
        for key, expected_shape in expected_shapes.items():
            self.assertEqual(np.asarray(experiment.demo_buffer[key]).shape, expected_shape)

    def test_learned_supervisor_actions_are_clamped_to_vmas_range(self):
        config = load_config("balance")
        config.update({"expert_pth": "heuristic", "horizon": 1, "seed": 0})
        experiment = R2BCExperiment(config)

        class OutOfRangePolicy:
            def compute_action(self, observation):
                return torch.tensor([[1.2, -1.3, 2.0, -2.0, 0.5, -0.5]])

        experiment.expert_pth = "checkpoint"
        experiment.expert_policy = OutOfRangePolicy()
        action = experiment._expert_action(torch.zeros(1, experiment.in_dims))
        self.assertTrue(torch.all(action <= 1.0))
        self.assertTrue(torch.all(action >= -1.0))

    def test_sparse_reward_adapter_preserves_multi_environment_reward_shape(self):
        config = load_config("balance")
        config.update({"expert_pth": "heuristic", "seed": 0})
        experiment = R2BCExperiment(config)
        env = experiment._make_env(num_envs=4, seed=0)
        env.reset(seed=0)
        _, rewards, _, _ = env.step([torch.zeros(4, 2) for _ in range(config["n"])])
        self.assertEqual([tuple(reward.shape) for reward in rewards], [(4,)] * config["n"])

    def test_buzz_wire_sparse_success_is_binary_and_terminal(self):
        from vmas import make_env

        scenario = SparseRewardBuzzWireScenario()
        env = make_env(
            scenario=scenario,
            num_envs=3,
            continuous_actions=True,
            max_steps=200,
            seed=0,
        )
        env.reset(seed=0)
        scenario.ball.state.pos[:] = scenario.goal.state.pos

        rewards = [scenario.reward(agent) for agent in scenario.world.agents]
        done = scenario.done()

        for reward in rewards:
            torch.testing.assert_close(reward, torch.ones(3))
        self.assertTrue(done.all())

    def test_balance_sparse_success_uses_native_overlap_and_is_terminal(self):
        from vmas import make_env

        scenario = SparseRewardBalanceScenario()
        env = make_env(
            scenario=scenario,
            num_envs=3,
            continuous_actions=True,
            max_steps=300,
            seed=0,
        )
        env.reset(seed=0)
        scenario.package.state.pos[:] = scenario.package.goal.state.pos

        rewards = [scenario.reward(agent) for agent in scenario.world.agents]
        done = scenario.done()

        for reward in rewards:
            torch.testing.assert_close(reward, torch.ones(3))
        self.assertTrue(done.all())

    def test_evaluation_counts_only_first_terminal_success_reward(self):
        class FakeEnv:
            def __init__(self):
                self.step_index = 0

            def reset(self, seed=None):
                return [torch.zeros(2, 3), torch.zeros(2, 3)]

            def step(self, actions):
                self.step_index += 1
                obs = [torch.zeros(2, 3), torch.zeros(2, 3)]
                if self.step_index == 1:
                    reward = torch.tensor([1.0, 0.0])
                    done = torch.tensor([True, False])
                else:
                    # Env 0 remains on-goal, but its post-terminal reward must
                    # not be counted a second time.
                    reward = torch.tensor([1.0, 1.0])
                    done = torch.tensor([True, True])
                return obs, [reward, reward.clone()], done, [{}, {}]

        experiment = object.__new__(R2BCExperiment)
        experiment.horizon = 5
        experiment.n_agents = 2
        experiment.device = torch.device("cpu")
        experiment.policy = torch.nn.Linear(6, 4)
        torch.nn.init.zeros_(experiment.policy.weight)
        torch.nn.init.zeros_(experiment.policy.bias)
        experiment.loss_fn = torch.nn.MSELoss()
        experiment._make_env = lambda num_envs, seed: FakeEnv()
        experiment._expert_action = lambda obs: torch.zeros(2, 2, 1, 2)

        average_reward, _ = experiment.test_batched_rollouts(n_evals=2)
        self.assertEqual(average_reward, 1.0)


if __name__ == "__main__":
    unittest.main()
