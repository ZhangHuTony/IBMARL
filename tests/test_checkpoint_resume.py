import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.run_experiment import create_run_dirs, load_resume_config
from src.util.checkpointing import ResumeMixin
from src.util.metrics_logger import MetricsLogger


class _Stateful:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class _Buffer:
    def __init__(self, value):
        self.value = value

    def dumps(self, path):
        torch.save(self.value, Path(path) / "value.pt")

    def loads(self, path):
        self.value = torch.load(
            Path(path) / "value.pt", map_location="cpu", weights_only=False
        )


class _Collector:
    total_frames = 0


class _Experiment(ResumeMixin):
    def __init__(self, root):
        self.config = {
            "check_dir": str(root / "checkpoints"),
            "frames_per_batch": 10,
            "n_iters": 20,
        }
        self.device = torch.device("cpu")
        self.metrics_logger = MetricsLogger(root / "data" / "metrics.csv")
        self.module = _Stateful(3)
        self.buffer = _Buffer([1, 2, 3])
        self.collector = _Collector()

    def _resume_modules(self):
        return {"module": self.module}

    def _resume_buffers(self):
        return {"buffer": self.buffer}


class TestCheckpointResume(unittest.TestCase):
    def test_full_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = _Experiment(Path(directory))
            exp.metrics_logger.log(4, "agents", reward=1.25)
            random.seed(12)
            np.random.seed(12)
            torch.manual_seed(12)

            exp.save_resume(4, {"total_frames": 50})
            expected = (random.random(), np.random.rand(), torch.rand(1))

            exp.module.value = 99
            exp.buffer.value = []
            exp.metrics_logger.restore([], [])
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)

            self.assertEqual(exp.load_resume(), 5)
            actual = (random.random(), np.random.rand(), torch.rand(1))
            self.assertEqual(exp.module.value, 3)
            self.assertEqual(exp.buffer.value, [1, 2, 3])
            self.assertEqual(
                exp.metrics_logger.get_values("reward", "agents"), [1.25]
            )
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_final_checkpoint_is_kept_at_non_interval_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = _Experiment(Path(directory))
            exp.save_final_resume(6, {"total_frames": 70})
            state = torch.load(
                exp._resume_state_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(state["iteration"], 6)

    def test_run_copies_training_inputs_into_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            teacher = tmp_path / "teacher.pt"
            demos = tmp_path / "demos.pt"
            teacher.write_bytes(b"teacher")
            demos.write_bytes(b"demonstrations")
            cfg = {
                "exp_type": "ibmarl",
                "scenario_name": "balance",
                "r2bc_checkpoint_path": str(teacher),
                "demonstrations_path": str(demos),
            }

            old_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                saved = create_run_dirs(cfg)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(
                Path(saved["r2bc_checkpoint_path"]).read_bytes(), b"teacher"
            )
            self.assertEqual(
                Path(saved["demonstrations_path"]).read_bytes(), b"demonstrations"
            )
            manifest = yaml.safe_load(
                (tmp_path / saved["run_dir"] / "artifacts.yaml").read_text()
            )
            self.assertEqual(
                manifest["demonstrations_path"]["size_bytes"],
                len(b"demonstrations"),
            )

    def test_model_only_run_is_rejected_as_not_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "old_run"
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "config.yaml").write_text("n_iters: 1000\n")
            (run_dir / "checkpoints" / "policy_checkpoint.pt").write_bytes(
                b"model"
            )

            with self.assertRaisesRegex(
                FileNotFoundError, "cannot be resumed exactly"
            ):
                load_resume_config(str(run_dir), 1500)

    def test_resume_updates_total_target_and_preserves_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            state_dir = run_dir / "checkpoints" / "resume"
            state_dir.mkdir(parents=True)
            config = {
                "n_iters": 1000,
                "frames_per_batch": 30,
                "run_dir": "old-relative-path",
            }
            (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
            torch.save({"iteration": 999}, state_dir / "state.pt")

            resumed = load_resume_config(str(run_dir), 1500)

            self.assertEqual(resumed["n_iters"], 1500)
            self.assertEqual(resumed["total_frames"], 45_000)
            self.assertEqual(resumed["run_dir"], str(run_dir.resolve()))
            self.assertEqual(
                resumed["resume_history"][-1]["checkpoint_iteration"], 999
            )


if __name__ == "__main__":
    unittest.main()
