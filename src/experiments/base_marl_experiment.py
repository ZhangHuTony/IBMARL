"""
Base class for multi-agent reinforcement learning experiments.
"""

import contextlib
import random

import torch
from abc import abstractmethod
import numpy as np

from tensordict.nn import TensorDictSequential
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

from src.environment.make_env import make_env
from src.util.metrics_logger import MetricsLogger
from src.util.checkpointing import ResumeMixin

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@contextlib.contextmanager
def vmas_rng_guard():
    """
    Make a block of environment interaction invisible to every *other* VMAS
    environment in the process.

    VMAS versions with ``Environment.vmas_random_state`` keep a single
    class-level state that their ``local_seed`` decorator swaps into the global
    torch/numpy/random generators around environment calls.  VMAS 1.4.x uses
    those global generators directly.  In both cases every ``VmasEnv`` in the
    process therefore draws from shared state: the training env, the evaluation
    env and the rendering env alike.

    The practical consequence is that evaluation rollouts advance the stream the
    collector's ``reset_world_at`` spawns come from -- evaluating changes what is
    subsequently trained on.  Wrapping the evaluation block in this guard
    restores the shared state afterwards, so evaluation genuinely cannot reach
    training and its cost in perturbation does not scale with how many protocols
    are measured.

    For the newer implementation the restore must be an in-place slice
    assignment: ``local_seed`` closed over the list object at class-definition
    time, so rebinding the attribute would not be seen.
    """
    from vmas.simulator.environment.environment import Environment

    state = getattr(Environment, "vmas_random_state", None)
    if state is not None:
        snapshot = [state[0].clone(), state[1], state[2]]
        try:
            yield
        finally:
            state[:] = snapshot
        return

    # VMAS 1.4.x seeds and consumes the process-wide generators directly.
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class BaseMARLExperiment(ResumeMixin):

    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._setup_seed()

        self.env = make_env(config, self.device)
        self._eval_env = None
        self.render_path = config['videos_dir']
        self.experiment_type = config['exp_type']

        data_dir = Path(config["data_dir"])
        self.metrics_logger = MetricsLogger(data_dir / "metrics.csv")

    @property
    def eval_env(self):
        """
        Dedicated environment for evaluation rollouts and video rendering.

        VMAS environments are stateful, so rolling out ``self.env`` -- the env
        the SyncDataCollector is driving -- resets its step counter and world
        state mid-run.  That desynchronises the collector from the simulator:
        the batch collected after an eval starts from a stale cached
        observation, and its first frame fires ``done`` immediately, logging a
        1-step episode (the ``episode_reward_mean = -1.0`` spike).  Evaluating
        in a separate env leaves the collector's trajectory untouched.

        Seeded off the training seed so eval episodes are not drawn from the
        same reset sequence as the training episodes, while staying
        reproducible and identical across variants at the same seed.

        Note that the seed argument is the *only* thing separating the two.  All
        VMAS environments in a process share one RNG stream (see
        ``vmas_rng_guard``), so a rollout here still advances the stream the
        collector resets from unless the caller holds that guard.

        Built on first use so eval-only experiments that never drive a
        collector don't pay for a second simulator.
        """
        if self._eval_env is None:
            eval_config = dict(self.config)
            eval_config["seed"] = self.config.get("seed", 0) + 10_000
            self._eval_env = make_env(eval_config, self.device)
        return self._eval_env

    def _setup_device(self):
        device = (
            torch.device(0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return device

    def _setup_seed(self):
        self.seed = self.config.get('seed', None)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        print(f"Setting seed to: {self.config.get('seed', None)}")

    @staticmethod
    def _agent_done_mask(td, group: str, reference) -> torch.Tensor:
        """
        Per-agent episode-end mask shaped like *reference* (the episode_reward
        tensor, [..., n_agents, 1]).

        A raw ``env.rollout`` output only carries the global ``("next", "done")``
        key -- the per-group key is added by ``process_batch`` on collector
        batches, not here.  Reading ``("next", group, "done")`` off a rollout
        therefore returns None, and ``tensor[None]`` silently inserts an axis
        instead of masking, which is why this has to be built explicitly.
        """
        done = td.get(("next", group, "done"), None)
        if done is None:
            done = td.get(("next", "done"))
            n_agents = reference.shape[-2]
            done = done.unsqueeze(-2).expand(*done.shape[:-1], n_agents, 1)
        return done

    def should_evaluate(self, iteration: int) -> bool:
        """
        Whether the (0-based) *iteration* about to be logged should run a
        dedicated evaluation.

        Evaluation is the dominant per-iteration cost on slow scenarios
        (buzz_wire: ~64% of an iteration -- it rolls out twice the horizon of
        collection), and its result is only logged.  It is side-effect-free on
        training only when the caller wraps it in ``vmas_rng_guard``: the
        ``eval_env`` is a separate simulator but not a separate RNG stream, so an
        unguarded rollout shifts the collector's subsequent resets.  Thinning it
        out trades curve resolution for wall time.

        ``eval_interval`` defaults to 1, so navigation/balance/transport keep
        evaluating every iteration.  The final iteration always evaluates, so the
        tail-10 final-return statistic stays well defined.
        """
        interval = int(self.config.get("eval_interval", 1))
        if interval <= 1:
            return True
        last = int(self.config.get("n_iters", 0)) - 1
        return iteration % interval == 0 or iteration >= last

    def reseed_eval_env(self, seed: int) -> None:
        """
        Pin ``eval_env`` to *seed* so the next rollout starts from a known set of
        initial conditions.

        Used to make two evaluation protocols measured at the same iteration a
        *paired* comparison -- both see identical episodes, so their difference
        is not swamped by reset noise at 20 episodes.  Only meaningful inside
        ``vmas_rng_guard``, which contains the reseed.
        """
        self.eval_env.set_seed(int(seed))

    def evaluate(
        self,
        n_episodes: int = 20,
        policies: dict | None = None,
        extra_keys: tuple = (),
    ) -> dict:
        """
        Deterministic evaluation: run the learned policy (no exploration noise)
        and return the mean *completed-episode* return per group.

        Runs in ``self.eval_env``, never in ``self.env`` -- see the ``eval_env``
        docstring for why sharing the collector's env corrupts training data.

        The environment is vectorised over ``frames_per_batch // horizon`` VMAS
        sub-environments, so one rollout yields that many episodes; rollouts are
        repeated until at least *n_episodes* have finished.  Returns are recorded
        per agent (matching ``episode_reward_mean``), i.e. an episode with N
        agents contributes N samples to the mean.

        *policies* overrides what is rolled out; when omitted it falls back to
        self.rl_policies (IBMARL) or self.policies (MADDPG / RLFD / RFT).  IBMARL
        passes its arbitrated RL+IL policy here and its bare RL actors on a
        second call, which is how one iteration yields both protocols.

        *extra_keys* names per-group keys the policy writes into the rollout
        (e.g. ``arbiter_choice``).  Their means land in ``self.last_eval_extras``
        as ``{group: {key: float}}`` rather than in the return value, so the
        signature stays a plain ``{group: mean_return}`` for every caller.
        """
        if policies is None:
            policies = getattr(self, "rl_policies", None) or getattr(self, "policies", None)
        if policies is None:
            raise RuntimeError("No policies found (expected self.policies or self.rl_policies)")

        eval_policy = TensorDictSequential(*policies.values())
        eval_policy.eval()

        env = self.eval_env
        horizon = self.config.get("horizon", 100)
        n_envs = max(int(env.batch_size[0]) if len(env.batch_size) else 1, 1)
        n_rollouts = max(1, -(-n_episodes // n_envs))  # ceil

        returns = {group: [] for group in env.group_map.keys()}
        extras = {group: {key: [] for key in extra_keys} for group in env.group_map.keys()}
        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                for _ in range(n_rollouts):
                    # break_when_any_done=False is required for correctness, not
                    # just completeness.  The default stops the rollout at the
                    # FIRST sub-env to finish, and only episodes that have ended
                    # are counted below -- so a policy that solves the task has
                    # its slower episodes systematically discarded and scores
                    # too well.  Measured on ibmarl/seed_0: -19.42 from 87 of
                    # 630 agent-episodes, versus -21.46 from all 630.  It also
                    # restores the n_rollouts arithmetic, which assumes each
                    # rollout yields one episode per sub-env.
                    out = env.rollout(
                        horizon, policy=eval_policy, break_when_any_done=False
                    )
                    for group in env.group_map.keys():
                        ep_reward = out.get(("next", group, "episode_reward"))
                        done = self._agent_done_mask(out, group, ep_reward)
                        returns[group].append(ep_reward[done].float())
                        for key in extra_keys:
                            # Averaged over every executed step, not only the
                            # ones ending an episode: with
                            # break_when_any_done=False sub-envs auto-reset and
                            # keep stepping, so this matches the collector-side
                            # convention for the same quantity.
                            value = out.get((group, key), None)
                            if value is not None:
                                extras[group][key].append(value.float().reshape(-1))

        self.last_eval_extras = {
            group: {
                key: (torch.cat(chunks).mean().item() if chunks else None)
                for key, chunks in by_key.items()
            }
            for group, by_key in extras.items()
        }

        mean_reward_by_group = {}
        for group, chunks in returns.items():
            vals = torch.cat(chunks) if chunks else torch.empty(0)
            mean_reward_by_group[group] = (
                vals.mean().item() if vals.numel() > 0 else 0.0
            )
        return mean_reward_by_group

    def save_results(self):
        self.metrics_logger.save()
        print(f"Saved metrics to: {self.metrics_logger.path.resolve()}")
        self.save_checkpoint()
        self._save_rewards_plot()

    def _save_rewards_plot(self):
        data_dir = Path(self.config["data_dir"])
        plots_dir = Path(self.config.get("plots_dir", data_dir.parent / "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        groups = list(self.env.group_map.keys())
        has_eval = "eval_reward_mean" in self.metrics_logger.columns

        n_plots = 2 if has_eval else 1
        fig, axs = plt.subplots(
            len(groups), n_plots,
            figsize=(6 * n_plots, 4 * len(groups)),
            squeeze=False,
        )

        for i, group in enumerate(groups):
            iterations = self.metrics_logger.get_values("iteration", group=group)
            rewards = self.metrics_logger.get_values("episode_reward_mean", group=group)
            axs[i, 0].plot(iterations, rewards, label=f"In-batch reward ({group})")
            axs[i, 0].set_ylabel("Reward")
            axs[i, 0].set_title(f"{group}: Training batch")
            axs[i, 0].legend()
            axs[i, 0].grid(True, alpha=0.3)

            if has_eval:
                eval_vals = self.metrics_logger.get_values(
                    "eval_reward_mean", group=group
                )
                valid = [
                    (it, v)
                    for it, v in zip(iterations, eval_vals)
                    if v is not None
                ]
                if valid:
                    idxs, vals = zip(*valid)
                    axs[i, 1].plot(
                        idxs, vals,
                        label=f"Eval reward ({group})",
                        color="orange",
                    )
                axs[i, 1].set_ylabel("Reward")
                axs[i, 1].set_title(f"{group}: Eval (20 episodes)")
                axs[i, 1].legend()
                axs[i, 1].grid(True, alpha=0.3)

        axs[-1, 0].set_xlabel("Training iterations")
        if has_eval:
            axs[-1, 1].set_xlabel("Training iterations")

        plt.tight_layout()
        plot_path = plots_dir / "episode_rewards.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved rewards plot to: {plot_path.resolve()}")

    def save_checkpoint(self):
        policies = getattr(self, "rl_policies", None) or getattr(self, "policies", None)
        if policies is None:
            raise RuntimeError("No policies to save (expected self.policies or self.rl_policies)")

        check_dir = Path(self.config["check_dir"])
        check_dir.mkdir(parents=True, exist_ok=True)

        state_dicts = {group: policy.state_dict() for group, policy in policies.items()}
        checkpoint_path = check_dir / "policy_checkpoint.pt"
        torch.save(state_dicts, checkpoint_path)
        print(f"Saved policy checkpoint to: {checkpoint_path.resolve()}")

    def _record_policy_video(self, policies: dict, tag: str = "policy"):
        """
        Roll out *policies* deterministically in a rendering copy of the env and
        write ``<videos_dir>/<tag>.mp4`` plus ``<tag>.gif``.

        Shared by every experiment so the baselines and IBMARL produce videos
        under the same names. TorchRL's CSVLogger picks its own filename and
        nesting (currently ``vmas_logs/videos/<tag>_0.mp4``), so the finished
        file is located by taking the newest .mp4 under videos_dir rather than
        by matching a fixed pattern.

        Rendered in a *dedicated single-sub-env* simulator, not in ``self.env``
        or ``self.eval_env``:

        * Stepping the training env here would desynchronise the collector, the
          same way a shared eval env does (see the ``eval_env`` docstring).
        * ``rollout`` stops as soon as *any* sub-env is done, but
          ``PixelRenderTransform`` only ever draws sub-env 0.  On a vectorised
          env the recording therefore ended at the earliest finisher across all
          sub-envs while the agent on screen was still mid-episode -- which is
          why trained (goal-reaching) policies produced 13-25 frame videos
          while policies that never succeed always ran the full horizon.  With
          one sub-env the two coincide: the video ends exactly when the episode
          being watched ends.
        """
        import shutil

        import imageio.v3 as iio

        videos_dir = Path(self.config["videos_dir"])
        videos_dir.mkdir(parents=True, exist_ok=True)

        video_logger = CSVLogger(
            exp_name="vmas_logs", log_dir=str(videos_dir), video_format="mp4"
        )

        print(f"Creating rendering env for {tag} video...")
        horizon = self.config.get("horizon", 100)
        render_config = dict(self.config)
        render_config["seed"] = self.config.get("seed", 0) + 20_000
        render_config["frames_per_batch"] = horizon  # -> exactly one sub-env
        env_with_render = make_env(render_config, self.device)
        env_with_render = env_with_render.append_transform(
            PixelRenderTransform(
                out_keys=["pixels"],
                preproc=lambda x: x.copy(),
                as_non_tensor=True,
                mode="rgb_array",
            )
        )
        # skip=1: record every step.  The default (2) halves the frame rate,
        # which is harmless over a 100-step timeout but turns a 30-step success
        # into a 15-frame flash.
        env_with_render = env_with_render.append_transform(
            VideoRecorder(logger=video_logger, tag=tag, skip=1)
        )

        policy = TensorDictSequential(*policies.values())
        policy.eval()

        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                print(f"Rendering {tag} rollout...")
                env_with_render.rollout(horizon, policy=policy)

        env_with_render.transform.dump()

        final_mp4 = videos_dir / f"{tag}.mp4"
        candidates = [p for p in videos_dir.rglob("*.mp4") if p != final_mp4]
        mp4_path = (
            max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
        )

        if mp4_path is None or not mp4_path.exists():
            print(f"Warning: no mp4 produced for {tag}")
            return

        if mp4_path != final_mp4:
            shutil.copy(mp4_path, final_mp4)

        # A solved episode ends on the frame the agents arrive, which reads as a
        # cut-off clip even though it is complete.  Hold the last frame so the
        # final configuration is actually visible.
        hold = 20
        try:
            frames = list(iio.imread(str(final_mp4), index=None))
            frames = frames + [frames[-1]] * hold
            # Written to a sibling first: imageio cannot encode into a path it
            # is still holding open for reading.
            padded = final_mp4.with_suffix(".padded.mp4")
            iio.imwrite(str(padded), frames, fps=30, codec="libx264")
            padded.replace(final_mp4)
        except Exception as e:
            print(f"Warning: could not pad {tag} video ({e}); keeping unpadded")
            frames = None
        print(f"Saved {tag} video to: {final_mp4.resolve()}")

        gif_path = videos_dir / f"{tag}.gif"
        try:
            if frames is None:
                frames = list(iio.imread(str(final_mp4), index=None))
            step = max(1, len(frames) // 60)
            iio.imwrite(
                str(gif_path), frames[::step], duration=step / 30.0, loop=0
            )
            print(f"Saved {tag} GIF to: {gif_path.resolve()}")
        except Exception as e:
            print(f"Warning: Could not create GIF for {tag}: {e}")

    def render_policy(self):
        raise NotImplementedError

    @abstractmethod
    def train(self) -> str:
        ...
