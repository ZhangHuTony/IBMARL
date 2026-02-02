
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




    def train(self):
        print("Training IBMARL Experiment...")

        pbar = tqdm(
            total= self.config.get('n_iters'),
            desc = ", ".join(
                [f"episode_reward_mean_{group}=0" for group in self.env.group_map.keys()]
            ), 
        )

        episode_reward_mean_map = {group: [] for group in self.env.group_map.keys()}
        rl_action_fraction_map = {group: [] for group in self.env.group_map.keys()}
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

    
    def save_checkpoint(self):
        raise NotImplementedError
    
    def render_policy(self):
        
        results_dir = self.render_path


        video_logger = CSVLogger(
            exp_name="vmas_logs",
            log_dir=str(results_dir),
            video_format="mp4",
        )

        print("Creating rendering env")
        env_with_render = TransformedEnv(self.env.base_env, self.env.transform.clone())

        env_with_render = env_with_render.append_transform(
            PixelRenderTransform(
                out_keys=["pixels"],
                preproc=lambda x: x.copy(),  # fix negative stride issue
                as_non_tensor=True,
                mode="rgb_array",
            )
        )

        env_with_render = env_with_render.append_transform(
            VideoRecorder(logger=video_logger, tag="vmas_rendered")
        )

        # deterministic policy (no exploration noise)
        render_policy = TensorDictSequential(*self.rl_policies.values())
        
        render_policy.eval()

        with torch.no_grad():
            with set_exploration_type(ExplorationType.MODE):
                print("Rendering rollout...")
                env_with_render.rollout(100, policy=render_policy)

        print("Saving video...")
        env_with_render.transform.dump()

        print("Saved! Video location:")
        video_logger.print_log_dir()
    
 


    




    

    




   