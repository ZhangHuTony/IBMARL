
from src.experiments.base_marl_experiment import BaseMARLExperiment




import copy
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
from src.experiments.ibmarl.data import build_data_collector, build_replay_buffer, process_batch


class IbmarlExperiment(BaseMARLExperiment):
    def __init__(self, config):
        super().__init__(config)


        #setup networks
        bc_path = Path(config["r2bc_checkpoint_path"])
        self.il_policies = R2bcPolicy(bc_path, self.env, self.device)

        self.rl_policies, self.rl_noise_policies = build_rl_policies(config, self.env, self.device)

        self.critics = build_critics(config, self.env, self.device)

        self.target_policies, self.target_critics = build_targets(self.rl_policies, self.critics, self.env)


        self.action_arbiter = ActionArbiter(config, self.il_policies, self.rl_policies, self.target_policies, self.critics, self.target_critics, self.env, self.device)

        self.agents_exploration_policy, self.collector = build_data_collector(config, self, self.rl_noise_policies, self.env, self.device)

        self.replay_buffers = build_replay_buffer(config, self.env, self.device)


        self.trainer = GroupTrainer(config, self.rl_policies, self.critics, self.target_policies, self.target_critics, self.action_arbiter, self.env)
        self._rl_video_created = False

    def evaluate_rl_only(self, n_episodes: int = 10) -> dict:
        """
        Evaluate only the RL policies (no IL / arbiter) in the environment.
        Runs rollouts until n_episodes are completed, then returns mean episode reward per group.
        """
        rl_only_policy = TensorDictSequential(*self.rl_policies.values())
        rl_only_policy.eval()

        horizon = self.config.get("horizon", 100)
        max_steps = horizon * (n_episodes + 5)

        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                out = self.env.rollout(max_steps, policy=rl_only_policy)

        mean_reward_by_group = {}
        for group in self.env.group_map.keys():
            done = out.get(("next", group, "done"))
            episode_rewards = out.get(("next", group, "episode_reward"))[done]
            if episode_rewards.numel() == 0:
                mean_reward_by_group[group] = 0.0
                continue
            # Use up to n_episodes (first n_episodes) and average
            n = min(n_episodes, episode_rewards.shape[0])
            mean_reward_by_group[group] = episode_rewards[:n].float().mean().item()

        return mean_reward_by_group

    def train(self) -> str:
        print("Training IBMARL Experiment...")

        pbar = tqdm(
            total= self.config.get('n_iters'),
            desc = ", ".join(
                [f"episode_reward_mean_{group}=0" for group in self.env.group_map.keys()]
            ), 
        )

        episode_reward_mean_map = {group: [] for group in self.env.group_map.keys()}
        rl_action_fraction_map = {group: [] for group in self.env.group_map.keys()}
        rl_only_episode_reward_mean_map = {group: [] for group in self.env.group_map.keys()}
        train_group_map = copy.deepcopy(self.env.group_map)

        for iteration, batch in enumerate(self.collector):

            for group in self.env.group_map.keys():
                # Retrieve the choices saved in modules.py
                choices = batch.get((group, "arbiter_choice")) 
                
                if choices is not None:
                    frac = choices.float().mean().item()
                    rl_action_fraction_map[group].append(frac)
                else:
                    print(f"Warning: No arbiter choice found for {group}")
                    
            current_frames = batch.numel()
            batch = process_batch(self.env, batch)


            for group in train_group_map.keys():
                group_batch = batch.exclude(
                    *[
                        key 
                        for _group in self.env.group_map.keys()
                        if _group != group
                        for key in [_group, ("next", _group)]
                    ]
                ) #exclude other groups' data
                group_batch = group_batch.reshape(
                    -1
                ) 

                # 1. Force unlock the internal storage of the replay buffer
                # This is often necessary if the buffer was created before the collector
                if hasattr(self.replay_buffers[group].storage, "_storage"):
                    if isinstance(self.replay_buffers[group].storage._storage, TensorDict):
                        self.replay_buffers[group].storage._storage.unlock_()

                # 2. Extend with a clone to be safe
                self.replay_buffers[group].extend(group_batch.clone())



                for _ in range(self.config.get('training').get('n_optimiser_steps')):

                    for _ in range(self.config.get('training').get('num_critic_updates')):
                        minibatch = self.replay_buffers[group].sample()
                        self.trainer.update_critic(group, minibatch) #TODO: save returns
                        self.trainer.polyak_step(self.critics[group], self.target_critics[group])

                    self.trainer.update_actor(group, minibatch) #TODO: save returns
                    self.trainer.polyak_step(self.rl_policies[group], self.target_policies[group])


                    # Annealing update for exploration noise
                self.rl_noise_policies[group][-1].step(current_frames)

            # On occasion, evaluate ONLY THE RL part of the IBMARL policy on 10 episodes in the environment.
            if iteration % 1 == 0:
                rl_only_means = self.evaluate_rl_only(n_episodes=10)
                for group in self.env.group_map.keys():
                    rl_only_episode_reward_mean_map[group].append(rl_only_means[group])
            else:
                for group in self.env.group_map.keys():
                    rl_only_episode_reward_mean_map[group].append(None)

            # Logging
            for group in self.env.group_map.keys():
                episode_reward_mean = (
                    batch.get(("next", group, "episode_reward"))[
                        batch.get(("next", group, "done"))
                    ]
                    .mean()
                    .item()
                    )
                
                episode_reward_mean_map[group].append(episode_reward_mean)

            pbar.set_description(
                ", ".join(
                    [
                        f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]}"
                        for group in self.env.group_map.keys()
                    ]
                ),
                refresh=False
            )
            pbar.update()

        self.results["group_map_keys"] = self.env.group_map.keys()
        self.results["episode_reward_mean_map"] = episode_reward_mean_map
        self.results["rl_action_fraction"] = rl_action_fraction_map
        self.results["rl_only_episode_reward_mean_map"] = rl_only_episode_reward_mean_map

        return f"IBMARL training complete. Environment: {self.config['scenario_name']}, Experiment Type: {self.experiment_type}, Seed: {self.seed}.\n\nResults: {self.results['rl_only_episode_reward_mean_map']['agents'][-10:]}"

    def save_results(self):
        """Override to add RL policy video/gif creation for IBMARL."""
        super().save_results()
        self._create_rl_policy_video_and_gif()

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

        # Find the created mp4 file (CSVLogger saves to log_dir/exp_name or log_dir)
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
            # Save a copy with a deterministic name
            final_mp4 = videos_dir / "rl_policy.mp4"
            if mp4_path != final_mp4:
                import shutil
                shutil.copy(mp4_path, final_mp4)
            print(f"Saved RL policy video to: {final_mp4.resolve()}")

            # Convert to GIF
            gif_path = videos_dir / "rl_policy.gif"
            try:
                frames = iio.imread(str(final_mp4), index=None)
                # Subsample frames for smaller gif (e.g. every 2nd frame)
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
        self._create_rl_policy_video_and_gif()  # Skips if already created by save_results
    
 


    




    

    




   