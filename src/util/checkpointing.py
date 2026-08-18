"""
Mid-run checkpoint / resume support.

A long sweep can be interrupted (reboot, machine needed elsewhere, Ctrl-C).
Every experiment periodically writes a *resume state* into
``<check_dir>/resume/`` holding everything needed to continue:

    state.pt        iteration, counters, RNG states, module/optimiser state dicts
    buffer_<name>/  a TorchRL ReplayBuffer dump per buffer

On the next launch the experiment reloads that state and continues from the
next iteration instead of starting over.  The resume state is deleted once a
run finishes normally, so only in-flight runs occupy disk.

What is *not* restored is the collector's environment state: the resume point
is a batch boundary, and with ``frames_per_batch == horizon * num_envs`` each
batch is a whole episode per sub-environment, so a fresh collector starting at
a reset is distributionally equivalent.  Networks, optimisers, target networks,
replay buffers, exploration-noise schedules and RNG streams are all preserved.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class ResumeMixin:
    """Mixin for BaseMARLExperiment subclasses providing resume support."""

    # ------------------------------------------------------------------
    # to be provided by the experiment
    # ------------------------------------------------------------------
    def _resume_modules(self) -> Dict[str, Any]:
        """name -> object exposing state_dict()/load_state_dict()."""
        return {}

    def _resume_buffers(self) -> Dict[str, Any]:
        """name -> TorchRL ReplayBuffer."""
        return {}

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    @property
    def resume_dir(self) -> Path:
        return Path(self.config["check_dir"]) / "resume"

    @property
    def _resume_state_path(self) -> Path:
        return self.resume_dir / "state.pt"

    @property
    def resume_interval(self) -> int:
        return int(self.config.get("resume_interval", 25))

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save_resume(self, iteration: int, counters: Dict[str, Any]) -> None:
        """Persist enough state to continue after *iteration* completed."""
        tmp_dir = self.resume_dir.with_name(self.resume_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for name, buf in self._resume_buffers().items():
            buf_dir = tmp_dir / f"buffer_{name}"
            buf_dir.mkdir(parents=True, exist_ok=True)
            buf.dumps(buf_dir)

        state = {
            "iteration": int(iteration),
            "counters": dict(counters),
            "modules": {
                name: obj.state_dict() for name, obj in self._resume_modules().items()
            },
            "rng": {
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "numpy": np.random.get_state(),
            },
            "metrics_rows": self.metrics_logger.rows,
            "metrics_columns": self.metrics_logger.columns,
        }
        torch.save(state, tmp_dir / "state.pt")

        # Swap in atomically-ish: a crash mid-write leaves the previous resume
        # point intact rather than a half-written one.
        if self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)
        tmp_dir.rename(self.resume_dir)
        print(f"[resume] saved at iteration {iteration} -> {self.resume_dir}")

    def load_resume(self) -> int:
        """
        Restore state if a resume point exists.

        Returns the iteration to start from (0 when there is nothing to resume).
        """
        if not self._resume_state_path.exists():
            return 0

        state = torch.load(
            self._resume_state_path, map_location=self.device, weights_only=False
        )

        modules = self._resume_modules()
        for name, obj in modules.items():
            if name in state["modules"]:
                obj.load_state_dict(state["modules"][name])
            else:
                print(f"[resume] warning: no saved state for module {name!r}")

        for name, buf in self._resume_buffers().items():
            buf_dir = self.resume_dir / f"buffer_{name}"
            if buf_dir.exists():
                buf.loads(buf_dir)
            else:
                print(f"[resume] warning: no saved dump for buffer {name!r}")

        rng = state.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            try:
                # state.pt is loaded with map_location=self.device, which puts
                # these ByteTensors on the GPU; set_rng_state_all rejects
                # anything that is not a CPU ByteTensor.
                torch.cuda.set_rng_state_all(
                    [s.cpu().to(torch.uint8) for s in rng["torch_cuda"]]
                )
            except Exception as e:  # differing device count between runs
                print(f"[resume] could not restore CUDA RNG state: {e}")

        self.metrics_logger.restore(
            state.get("metrics_rows", []), state.get("metrics_columns", [])
        )

        self._resume_counters = dict(state.get("counters", {}))
        start_iteration = int(state["iteration"]) + 1
        print(
            f"[resume] restored from iteration {state['iteration']}; "
            f"continuing at {start_iteration}"
        )
        return start_iteration

    def clear_resume(self) -> None:
        """Drop the resume state (called after a run completes normally)."""
        for d in (self.resume_dir, self.resume_dir.with_name(self.resume_dir.name + ".tmp")):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------
    # helpers used by train()
    # ------------------------------------------------------------------
    def begin_training(self, default_counters: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """
        Restore any resume point and return (start_iteration, counters).

        Also shrinks the collector's frame budget to what is left, so the
        already-collected iterations are not repeated.
        """
        start_iteration = self.load_resume()
        counters = dict(default_counters)
        if start_iteration > 0:
            counters.update(getattr(self, "_resume_counters", {}))
            n_iters = int(self.config["n_iters"])
            remaining = max(0, n_iters - start_iteration)
            self.collector.total_frames = (
                int(self.config["frames_per_batch"]) * remaining
            )
            print(
                f"[resume] collector budget set to {self.collector.total_frames} "
                f"frames ({remaining} iterations)"
            )
        return start_iteration, counters
