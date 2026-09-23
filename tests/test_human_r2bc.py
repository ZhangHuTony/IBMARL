"""Unit tests for the human-R2BC artifact boundary (no controller required)."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.r2bc.human_teleop import DemonstrationBuffer, load_human_config
from src.r2bc.human_gamepad import XboxGamepad
from src.r2bc.mabc import DecentralizedMiniBC


class TestHumanR2BC(unittest.TestCase):
    def test_human_buzz_wire_uses_local_sparse_binary_reward(self):
        config = load_human_config("buzz_wire")
        self.assertTrue(config["sparse_rewards"])
        self.assertTrue(config["binary_terminal_reward"])

    def test_xbox_axes_align_with_vmas_viewer(self):
        # Build without initializing pygame or requiring controller hardware.
        gamepad = object.__new__(XboxGamepad)
        gamepad.deadzone = 0.1
        gamepad.invert_x = False
        gamepad.invert_y = False
        gamepad.swap_axes = False

        # pygame: right=(+1, 0), up=(0, -1). These corrected VMAS actions
        # display as right and up respectively.
        torch.testing.assert_close(
            torch.from_numpy(gamepad.map_stick(1.0, 0.0)),
            torch.tensor([1.0, 0.0]),
        )
        torch.testing.assert_close(
            torch.from_numpy(gamepad.map_stick(0.0, -1.0)),
            torch.tensor([0.0, 1.0]),
        )

    def test_demonstrations_match_ibmarl_schema(self):
        buffer = DemonstrationBuffer()
        buffer.add_transition(
            torch.zeros(2, 6),
            torch.zeros(2, 2),
            torch.ones(2, 1),
            torch.ones(2, 6),
            True,
            True,
            1,
        )
        result = buffer.as_dict("buzz_wire")
        self.assertEqual(set(result) - {"meta"}, {
            "obs", "act", "rewards", "next_obs", "dones", "terminated"
        })
        self.assertEqual(tuple(result["obs"].shape), (1, 2, 6))
        self.assertEqual(tuple(result["act"].shape), (1, 2, 2))
        self.assertEqual(tuple(result["rewards"].shape), (1, 2, 1))
        self.assertEqual(result["meta"]["controlled_agent"].tolist(), [1])

    def test_demonstrations_can_be_restored_for_resume(self):
        buffer = DemonstrationBuffer()
        buffer.add_transition(
            torch.arange(12).reshape(2, 6),
            torch.arange(4).reshape(2, 2),
            torch.ones(2, 1),
            torch.ones(2, 6),
            False,
            False,
            1,
        )
        buffer.episodes.append({"episode": 0, "controlled_agent": 1, "length": 1})

        restored = DemonstrationBuffer.from_dict(buffer.as_dict("buzz_wire", "sparse"))

        self.assertEqual(restored.controlled_agent, [1])
        self.assertEqual(restored.episodes, buffer.episodes)
        torch.testing.assert_close(restored.obs[0], buffer.obs[0])
        torch.testing.assert_close(restored.act[0], buffer.act[0])

    def test_resume_requires_round_robin_metadata(self):
        with self.assertRaisesRegex(ValueError, "controlled-agent/episode metadata"):
            DemonstrationBuffer.from_dict({
                "obs": torch.zeros(1, 2, 6),
                "act": torch.zeros(1, 2, 2),
                "rewards": torch.zeros(1, 2, 1),
                "next_obs": torch.zeros(1, 2, 6),
                "dones": torch.zeros(1, dtype=torch.bool),
            })

    def test_checkpoint_loads_with_existing_policy_class(self):
        policy = DecentralizedMiniBC(2, 12, 4, hidden_size=16, hidden_layers=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_checkpoint.pth"
            policy.save_checkpoint(path)
            loaded = DecentralizedMiniBC.load_checkpoint(path, torch.device("cpu"))
        output = loaded(torch.zeros(3, 12))
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertEqual(loaded.hidden_layers, 2)


if __name__ == "__main__":
    unittest.main()
