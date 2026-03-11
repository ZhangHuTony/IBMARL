
from src.experiments.base_marl_experiment import BaseMARLExperiment


import copy
import time
import torch


from tqdm import tqdm

from pathlib import Path
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

from tensordict.nn import TensorDictSequential
from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tensordict import TensorDict


from src.experiments.ibmarl.networks import R2bcPolicy, build_rl_policies, build_critics, build_targets
from src.experiments.ibmarl.losses import GroupTrainer
from src.experiments.ibmarl.arbiter import ActionArbiter
from src.experiments.ibmarl.modules import build_il_noise_modules
from src.experiments.ibmarl.data import (
    build_data_collector,
    build_single_replay_buffer,
    process_batch,
)


class IbmarlExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)

        bc_path = Path(config["r2bc_checkpoint_path"])
        self.il_policies = R2bcPolicy(bc_path, self.env, self.device)

        self.rl_policies = build_rl_policies(config, self.env, self.device)

        self.critics = build_critics(config, self.env, self.device)

        self.target_policies, self.target_critics = build_targets(self.rl_policies, self.critics, self.env)

        self.il_noise_modules = build_il_noise_modules(config, self.env, self.device)
        self.action_arbiter = ActionArbiter(
            config,
            self.il_policies,
            self.rl_policies,
            self.target_policies,
            self.critics,
            self.target_critics,
            self.env,
            self.device,
            il_noise_modules=self.il_noise_modules,
        )

        self.agents_exploration_policy, self.collector, self.noise_modules = build_data_collector(config, self, self.rl_policies, self.env, self.device)

        # IBRL-style: single replay buffer with demos pre-loaded
        self.replay_buffers = build_single_replay_buffer(
            config, self.env, self.device
        )

        self.trainer = GroupTrainer(config, self.rl_policies, self.critics, self.target_policies, self.target_critics, self.action_arbiter, self.env)
        self._rl_video_created = False

    def train(self) -> str:
        print("Training IBMARL Experiment...")

        pbar = tqdm(
            total= self.config.get('n_iters'),
            desc = ", ".join(
                [f"episode_reward_mean_{group}=0" for group in self.env.group_map.keys()]
            ), 
        )

        train_group_map = copy.deepcopy(self.env.group_map)

        start_time = time.time()
        total_frames = 0
        total_episodes = 0
        total_train_steps = 0

        train_batch_size = self.config.get("training").get("train_batch_size")
        min_warm_up_frames = self.config.get("ibmarl_min_warm_up_frames", 5000)
        first_group = list(self.env.group_map.keys())[0]

        for iteration, batch in enumerate(self.collector):

            # --- Arbiter metrics from batch ---
            arbiter_metrics = {}
            for group in self.env.group_map.keys():
                choices = batch.get((group, "arbiter_choice"))

                rl_frac = choices.float().mean().item() if choices is not None else 0.0

                mean_action_diff = batch.get((group, "mean_action_diff"))
                mean_q_diff = batch.get((group, "mean_q_diff"))
                var_q_diff = batch.get((group, "var_q_diff"))

                arbiter_metrics[group] = {
                    "rl_action_fraction": rl_frac,
                    "mean_action_diff": mean_action_diff.mean().item() if mean_action_diff is not None else 0.0,
                    "mean_q_diff": mean_q_diff.mean().item() if mean_q_diff is not None else 0.0,
                    "var_q_diff": var_q_diff.mean().item() if var_q_diff is not None else 0.0,
                }

            current_frames = batch.numel()
            total_frames += current_frames
            batch = process_batch(self.env, batch)

            # --- Add transitions to single replay buffer ---
            for group in train_group_map.keys():
                group_batch = batch.select(
                    group,
                    "next",
                    "done",
                    "terminated",
                    strict=False,
                )
                # Keep only the keys that match the demo-initialized buffer schema
                group_batch = group_batch.select(
                    (group, "observation"),
                    (group, "action"),
                    (group, "episode_reward"),
                    ("next", group, "observation"),
                    ("next", group, "action"),
                    ("next", group, "episode_reward"),
                    ("next", group, "reward"),
                    ("next", group, "done"),
                    ("next", group, "terminated"),
                    ("next", "done"),
                    ("next", "terminated"),
                    "done",
                    "terminated",
                    strict=False,
                )
                group_batch = group_batch.reshape(-1)
                self.replay_buffers[group].extend(group_batch)

            # --- Warm-up: skip training until buffer has enough data ---
            is_warm_up = len(self.replay_buffers[first_group]) < min_warm_up_frames
            if is_warm_up:
                print(
                    f"[Warm-up] iteration {iteration}: "
                    f"buffer={len(self.replay_buffers[first_group])}/{min_warm_up_frames}"
                )

            # --- Training (skipped during warm-up) ---
            actor_losses = {group: [] for group in self.env.group_map.keys()}
            critic_losses = {group: [] for group in self.env.group_map.keys()}

            if not is_warm_up:
                for group in train_group_map.keys():
                    for _ in range(self.config.get("training").get("n_optimiser_steps")):
                        minibatch = self.replay_buffers[group].sample(
                            batch_size=train_batch_size
                        )

                        critic_info = self.trainer.update_critic(group, minibatch)
                        actor_info = self.trainer.update_actor(group, minibatch)

                        critic_losses[group].append(critic_info["critic_loss"])
                        actor_losses[group].append(actor_info["actor_loss"])

                        self.trainer.polyak_step(
                            self.critics[group][0], self.target_critics[group][0]
                        )
                        self.trainer.polyak_step(
                            self.rl_policies[group], self.target_policies[group]
                        )

                        self.noise_modules[group].step(current_frames)
                        self.il_noise_modules[group].step(current_frames)

                        total_train_steps += 1

            # --- Dedicated evaluation (skip during warm-up) ---
            eval_means = None
            if not is_warm_up:
                eval_means = self.evaluate(n_episodes=20)

            # --- Metrics ---
            elapsed = time.time() - start_time
            speed = total_frames / max(elapsed, 1e-6)

            done_global = batch.get(("next", "done"))
            episodes_this_iter = int(done_global.sum().item())
            total_episodes += episodes_this_iter

            for group in self.env.group_map.keys():
                done = batch.get(("next", group, "done"))
                ep_rewards = batch.get(("next", group, "episode_reward"))[done]
                episode_reward_mean = ep_rewards.mean().item() if ep_rewards.numel() > 0 else 0.0

                n_opt = max(len(actor_losses[group]), 1)
                eval_val = eval_means[group] if eval_means is not None else None

                self.metrics_logger.log(
                    iteration=iteration,
                    group=group,
                    elapsed_time=round(elapsed, 2),
                    episode=total_episodes,
                    step=total_frames,
                    train_step=total_train_steps,
                    speed_fps=round(speed, 2),
                    episode_reward_mean=episode_reward_mean,
                    eval_reward_mean=eval_val,
                    actor_loss=round(sum(actor_losses[group]) / n_opt, 6) if actor_losses[group] else None,
                    critic_loss=round(sum(critic_losses[group]) / n_opt, 6) if critic_losses[group] else None,
                    replay_size=len(self.replay_buffers[group]),
                    rl_action_fraction=round(arbiter_metrics[group]["rl_action_fraction"], 4),
                    rl_only_episode_reward_mean=eval_val,
                    mean_action_diff=round(arbiter_metrics[group]["mean_action_diff"], 6),
                    mean_q_diff=round(arbiter_metrics[group]["mean_q_diff"], 6),
                    var_q_diff=round(arbiter_metrics[group]["var_q_diff"], 6),
                )

            if iteration % 10 == 0:
                self.metrics_logger.save()

            pbar.set_description(
                ", ".join(
                    [
                        f"episode_reward_mean_{group} = {self.metrics_logger.get_values('episode_reward_mean', group)[-1]}"
                        for group in self.env.group_map.keys()
                    ]
                ),
                refresh=False
            )
            pbar.update()

        self.metrics_logger.save()

        first_group = list(self.env.group_map.keys())[0]
        recent_rl_only = self.metrics_logger.get_values("rl_only_episode_reward_mean", first_group)[-10:]
        return (
            f"IBMARL training complete. Environment: {self.config['scenario_name']}, "
            f"Experiment Type: {self.experiment_type}, Seed: {self.seed}.\n\n"
            f"Results: {recent_rl_only}"
        )

    def save_results(self):
        """Override to add RL policy video/gif creation for IBMARL."""
        super().save_results()
        try:
            self._create_rl_policy_video_and_gif()
        except Exception as e:
            print(f"Could not save RL policy video: {e}")

    def save_checkpoint(self):
        """Base implementation saves self.rl_policies to check_dir."""
        super().save_checkpoint()

    def _create_rl_policy_video_and_gif(self):
        """
        Create .mp4 and .gif of the RL policy only (NOT the combined IL/RL policy).
        Saves to videos_dir.
        """
        if self._rl_video_created:
            return
        self._rl_video_created = True

        import imageio.v3 as iio

        videos_dir = Path(self.config["videos_dir"])
        videos_dir.mkdir(parents=True, exist_ok=True)

        video_logger = CSVLogger(
            exp_name="vmas_logs",
            log_dir=str(videos_dir),
            video_format="mp4",
        )

        print("Creating rendering env for RL policy video...")
        env_with_render = TransformedEnv(self.env.base_env, self.env.transform.clone())
        env_with_render = env_with_render.append_transform(
            PixelRenderTransform(
                out_keys=["pixels"],
                preproc=lambda x: x.copy(),
                as_non_tensor=True,
                mode="rgb_array",
            )
        )
        env_with_render = env_with_render.append_transform(
            VideoRecorder(logger=video_logger, tag="rl_policy")
        )

        rl_policy = TensorDictSequential(*self.rl_policies.values())
        rl_policy.eval()

        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                print("Rendering RL policy rollout...")
                env_with_render.rollout(100, policy=rl_policy)

        print("Saving video...")
        env_with_render.transform.dump()

        mp4_patterns = [
            videos_dir / "vmas_logs" / "video_rl_policy_*.mp4",
            videos_dir / "video_rl_policy_*.mp4",
        ]
        mp4_path = None
        for pattern in mp4_patterns:
            matches = sorted(pattern.parent.glob(pattern.name))
            if matches:
                mp4_path = matches[-1]
                break
        if mp4_path is None:
            matches = list(videos_dir.rglob("video_*.mp4"))
            if matches:
                mp4_path = sorted(matches)[-1]

        if mp4_path is not None and mp4_path.exists():
            final_mp4 = videos_dir / "rl_policy.mp4"
            if mp4_path != final_mp4:
                import shutil
                shutil.copy(mp4_path, final_mp4)
            print(f"Saved RL policy video to: {final_mp4.resolve()}")

            gif_path = videos_dir / "rl_policy.gif"
            try:
                frames = iio.imread(str(final_mp4), index=None)
                step = max(1, len(frames) // 60)
                frames_sub = frames[::step]
                iio.imwrite(str(gif_path), frames_sub, duration=step / 30.0, loop=0)
                print(f"Saved RL policy GIF to: {gif_path.resolve()}")
            except Exception as e:
                print(f"Warning: Could not create GIF: {e}")
        else:
            print("Warning: Could not find saved mp4 file for GIF conversion")

    def render_policy(self):
        """Render the RL policy only (not combined IL/RL). Uses _create_rl_policy_video_and_gif."""
        try:
            self._create_rl_policy_video_and_gif()
        except Exception as e:
            print(f"Could not render RL policy: {e}")
