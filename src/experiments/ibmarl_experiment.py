
from src.experiments.base_marl_experiment import BaseMARLExperiment, vmas_rng_guard


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
from src.experiments.ibmarl.modules import build_eval_policies, build_il_noise_modules
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

        # The RL+IL policy IBMARL actually executes, minus exploration: the
        # critic picks greedily between the RL actor's action and the teacher's.
        # This is what `eval_reward_mean` measures (IBRL evaluates the
        # bootstrapped policy, not the RL actor alone); the bare `rl_policies`
        # are evaluated alongside it as the secondary protocol.  It shares
        # parameters with `rl_policies`, so it is deliberately absent from
        # `_resume_modules`.
        self.eval_policies = build_eval_policies(
            self.action_arbiter, self.rl_policies, self.env
        )

        # IBRL-style: single replay buffer with demos pre-loaded
        self.replay_buffers = build_single_replay_buffer(
            config, self.env, self.device
        )

        self.trainer = GroupTrainer(config, self.rl_policies, self.critics, self.target_policies, self.target_critics, self.action_arbiter, self.env)
        self._rl_video_created = False

    def _resume_modules(self):
        modules = {}
        for group in self.env.group_map.keys():
            modules[f"rl_policy_{group}"] = self.rl_policies[group]
            modules[f"critics_{group}"] = self.critics[group]
            modules[f"target_policy_{group}"] = self.target_policies[group]
            modules[f"target_critics_{group}"] = self.target_critics[group]
            modules[f"opt_actor_{group}"] = self.trainer.optimisers[group]["loss_actor"]
            modules[f"opt_value_{group}"] = self.trainer.optimisers[group]["loss_value"]
            modules[f"noise_{group}"] = self.noise_modules[group]
            modules[f"il_noise_{group}"] = self.il_noise_modules[group]
        return modules

    def _resume_buffers(self):
        return {f"replay_{g}": b for g, b in self.replay_buffers.items()}

    def train(self) -> str:
        print("Training IBMARL Experiment...")

        start_iteration, counters = self.begin_training(
            {
                "total_frames": 0,
                "total_episodes": 0,
                "total_train_steps": 0,
                "elapsed": 0.0,
            }
        )

        pbar = tqdm(
            total= self.config.get('n_iters'),
            initial=start_iteration,
            desc = ", ".join(
                [f"episode_reward_mean_{group}=0" for group in self.env.group_map.keys()]
            ),
        )

        train_group_map = copy.deepcopy(self.env.group_map)

        start_time = time.time()
        elapsed_offset = counters["elapsed"]
        elapsed = elapsed_offset
        total_frames = counters["total_frames"]
        total_episodes = counters["total_episodes"]
        total_train_steps = counters["total_train_steps"]

        train_batch_size = self.config.get("training").get("train_batch_size")
        min_warm_up_frames = self.config.get("ibmarl_min_warm_up_frames", 5000)
        first_group = list(self.env.group_map.keys())[0]

        # Who acts while the buffer is below the warm-up threshold.
        #   teacher -- IBRL's warm-up (num_warm_up_episode): the teacher's noised
        #              proposal is executed and nothing is trained, so the
        #              arbiter's first decision is made by a critic that has
        #              seen the teacher succeed.
        #   arbiter -- the historical default: the untrained critic arbitrates
        #              the first batches.  On a task the RL actor cannot solve
        #              alone that is a coin flip per seed (buzz_wire: 2/5
        #              ibmarl_strict seeds opened RL-heavy and never left the
        #              floor), so buzz_wire overrides it.
        warm_up_policy = self.config.get("ibmarl_warm_up_policy", "arbiter")
        if warm_up_policy not in ("teacher", "arbiter"):
            raise ValueError(
                "ibmarl_warm_up_policy must be 'teacher' or 'arbiter', "
                f"got {warm_up_policy!r}"
            )

        def _set_warm_up_acting():
            # Governs the *next* batch the collector produces: called before
            # the first one and after every buffer extension.  A resumed run
            # restores its buffer in begin_training, so it lands past warm-up.
            self.action_arbiter.force_il = (
                warm_up_policy == "teacher"
                and len(self.replay_buffers[first_group]) < min_warm_up_frames
            )

        _set_warm_up_acting()

        for offset, batch in enumerate(self.collector):
            iteration = start_iteration + offset

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
            _set_warm_up_acting()  # the next batch: teacher while still warming up
            if is_warm_up:
                print(
                    f"[Warm-up] iteration {iteration}: "
                    f"buffer={len(self.replay_buffers[first_group])}/{min_warm_up_frames} "
                    f"(acting: {warm_up_policy})"
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
                            self.critics[group], self.target_critics[group]
                        )
                        self.trainer.polyak_step(
                            self.rl_policies[group], self.target_policies[group]
                        )

                        total_train_steps += 1

                    # Anneal exploration noise once per iteration, not once per
                    # optimiser step -- stepping inside the loop above advanced the
                    # schedule by n_optimiser_steps * current_frames and saturated
                    # sigma after a single iteration. Matches maddpg.py.
                    self.noise_modules[group].step(current_frames)
                    self.il_noise_modules[group].step(current_frames)

            # --- Dedicated evaluation (skip during warm-up) ---
            #
            # Two protocols at the same iteration.  `eval_means` is the RL+IL
            # arbitrated policy -- what IBMARL executes, and what IBRL reports;
            # `rl_only_means` is the RL actors alone, kept because it stays
            # comparable with the baselines and with every run already on disk
            # (steps-to-teacher in particular reads it, since the arbitrated
            # curve reaches teacher level as soon as the critic can rank two
            # actions).
            #
            # Both are measured from the *same* eval-env seed so the RL+IL minus
            # RL-only difference is paired, and the whole block sits inside the
            # RNG guard so neither rollout shifts the collector's resets.
            eval_means = None
            rl_only_means = None
            eval_rl_fraction = {}
            if not is_warm_up and self.should_evaluate(iteration):
                with vmas_rng_guard():
                    eval_seed = self.config.get("seed", 0) + 10_000 + iteration
                    self.reseed_eval_env(eval_seed)
                    eval_means = self.evaluate(
                        n_episodes=20,
                        policies=self.eval_policies,
                        extra_keys=("arbiter_choice",),
                    )
                    eval_rl_fraction = {
                        group: extras.get("arbiter_choice")
                        for group, extras in self.last_eval_extras.items()
                    }
                    if self.config.get("eval_rl_only", True):
                        self.reseed_eval_env(eval_seed)
                        rl_only_means = self.evaluate(
                            n_episodes=20, policies=self.rl_policies
                        )

            # --- Metrics ---
            elapsed = elapsed_offset + (time.time() - start_time)
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
                rl_only_val = (
                    rl_only_means[group] if rl_only_means is not None else None
                )
                eval_frac = eval_rl_fraction.get(group)

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
                    rl_only_episode_reward_mean=rl_only_val,
                    eval_rl_action_fraction=(
                        round(eval_frac, 4) if eval_frac is not None else None
                    ),
                    # Discriminator for the analysis scripts: runs predating this
                    # column have RL-only in `eval_reward_mean`, as do the
                    # baselines, so the two are not interchangeable.
                    eval_protocol="rl_il",
                    mean_action_diff=round(arbiter_metrics[group]["mean_action_diff"], 6),
                    mean_q_diff=round(arbiter_metrics[group]["mean_q_diff"], 6),
                    var_q_diff=round(arbiter_metrics[group]["var_q_diff"], 6),
                )

            if iteration % 10 == 0:
                self.metrics_logger.save()

            if (iteration + 1) % self.resume_interval == 0:
                self.metrics_logger.save()
                self.save_resume(
                    iteration,
                    {
                        "total_frames": total_frames,
                        "total_episodes": total_episodes,
                        "total_train_steps": total_train_steps,
                        "elapsed": elapsed,
                    },
                )

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
        self.save_final_resume(
            int(self.config["n_iters"]) - 1,
            {
                "total_frames": total_frames,
                "total_episodes": total_episodes,
                "total_train_steps": total_train_steps,
                "elapsed": elapsed,
            },
        )

        first_group = list(self.env.group_map.keys())[0]

        def _tail(column):
            # Filter before slicing: MetricsLogger omits a key whose value is
            # None, so on a thinned eval schedule (buzz_wire's eval_interval: 4)
            # a raw [-10:] is mostly Nones and the summary reads as empty.
            values = [
                v
                for v in self.metrics_logger.get_values(column, first_group)
                if v is not None
            ]
            return values[-10:]

        return (
            f"IBMARL training complete. Environment: {self.config['scenario_name']}, "
            f"Experiment Type: {self.experiment_type}, Seed: {self.seed}.\n\n"
            f"RL+IL eval: {_tail('eval_reward_mean')}\n\n"
            f"RL-only eval: {_tail('rl_only_episode_reward_mean')}"
        )

    def save_results(self):
        """Override to add policy video/gif creation for IBMARL."""
        super().save_results()
        try:
            self._create_policy_videos_and_gifs()
        except Exception as e:
            print(f"Could not save policy videos: {e}")

    def save_checkpoint(self):
        """
        Base implementation saves self.rl_policies to check_dir; add the critics.

        The arbiter needs a critic to score the two proposals, so the RL+IL
        protocol cannot be re-measured from an actors-only checkpoint.  Both the
        online and the target ensembles are written: the arbiter scores with the
        targets, and keeping the online set means the online-vs-target scoring
        question can be answered post-hoc without a retrain.
        """
        super().save_checkpoint()

        check_dir = Path(self.config["check_dir"])
        check_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "critics": {g: c.state_dict() for g, c in self.critics.items()},
            "target_critics": {
                g: c.state_dict() for g, c in self.target_critics.items()
            },
            "num_critics": self.config.get("num_critics"),
            "strict": self.config.get("strict"),
            "arbiter_scores_with": "target",
        }
        critic_path = check_dir / "critic_checkpoint.pt"
        torch.save(payload, critic_path)
        print(f"Saved critic checkpoint to: {critic_path.resolve()}")

    def _create_policy_videos_and_gifs(self):
        """
        Create .mp4 and .gif for both evaluation protocols: the RL actors alone
        and the arbitrated RL+IL policy that `eval_reward_mean` measures.

        Wrapped in the RNG guard for the same reason evaluation is -- rendering
        steps a VMAS env, and all of them share one RNG stream.
        """
        if self._rl_video_created:
            return
        self._rl_video_created = True
        with vmas_rng_guard():
            self._record_policy_video(self.rl_policies, tag="rl_policy")
            self._record_policy_video(self.eval_policies, tag="rl_il_policy")

    def render_policy(self):
        """Render both the RL-only and the arbitrated RL+IL policy."""
        try:
            self._create_policy_videos_and_gifs()
        except Exception as e:
            print(f"Could not render policies: {e}")
