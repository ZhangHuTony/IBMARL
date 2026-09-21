
from tensordict import TensorDictBase
import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm

from src.experiments.base_marl_experiment import BaseMARLExperiment

from torchrl.modules import (
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
    AdditiveGaussianModule,
)

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.objectives import DDPGLoss, ValueEstimators, SoftUpdate

from tensordict.nn import TensorDictModule, TensorDictSequential
from pathlib import Path


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder


class MaddpgExperiment(BaseMARLExperiment):

    def __init__(self, config):
        super().__init__(config)

        self.policies, self.exploration_policies = self._setup_policy(self.config, self.env, self.device)

        self.critics = self._setup_critic()

        self.agents_exploration_policy, self.collector = self._setup_data_collection()

        self.replay_buffers = self._setup_replay_buffer()

        self.losses, self.target_updaters, self.optimisers = self._setup_loss_functions()


    def _setup_policy(self, cfg, env, device):
        print("Setting up Policies...")

        policy_modules = {}

        for group, agents in env.group_map.items():
            policy_net = MultiAgentMLP(
                n_agent_inputs= env.observation_spec[group, "observation"].shape[-1],
                n_agent_outputs= env.full_action_spec[group, "action"].shape[-1],
                n_agents = len(agents),
                centralized=False,
                share_params= False,
                device = device,
                depth = 2,
                num_cells = 256,
                activation_class= torch.nn.Tanh
            )

            policy_module = TensorDictModule(
                policy_net,
                in_keys=[(group, "observation")],
                out_keys=[(group, "param")],
            )

            policy_modules[group] = policy_module
            
        policies = {}

        for group, _agents in env.group_map.items():
            low = env.full_action_spec_unbatched[group, "action"].space.low.to(device)
            high = env.full_action_spec_unbatched[group, "action"].space.high.to(device)

            policy = ProbabilisticActor(
                module=policy_modules[group],
                spec=env.full_action_spec[group, "action"],
                in_keys=[(group, "param")],
                out_keys=[(group, "action")],
                distribution_class=TanhDelta,
                distribution_kwargs={"low": low, "high": high},
                return_log_prob=False,
            )

            policies[group] = policy
        
        # Read from the same `exploration_noise` block IBMARL reads
        # (ibmarl/modules.py).  The values were hard-coded here at 0.1/0.1,
        # numerically identical to ibmarl.yaml's defaults, so nothing changes
        # today -- but a sigma sweep (paper_run --sigma-init/--sigma-end, or an
        # edit to the yaml) moved IBMARL's exploration and silently left every
        # baseline behind at 0.1.  Same source, same schedule, one knob.
        noise_config = cfg.get('exploration_noise') or {}
        sigma_init = noise_config.get('sigma_init', 0.1)
        sigma_end = noise_config.get('sigma_end', 0.1)

        exploration_policies = {}
        for group, _agents in env.group_map.items():
            exploration_policy = TensorDictSequential(
                policies[group],
                AdditiveGaussianModule(
                    spec = policies[group].spec,
                    annealing_num_steps=cfg.get(
                        "exploration_annealing_frames", cfg.get("total_frames") // 2
                    ),
                    action_key= (group, "action"),
                    sigma_init = sigma_init,
                    sigma_end = sigma_end,
                )
            )
            exploration_policies[group] = exploration_policy
        
        return policies, exploration_policies
        
    def _setup_critic(self):
        critics = {}

        share_critic_params = False
        centralized = True

        print("Setting up MADDPG critic networks...")
        for group, agents in self.env.group_map.items():
            
            cat_module = TensorDictModule(
                lambda obs, action: torch.cat([obs, action], dim=-1),
                in_keys=[(group, "observation"), (group, "action")],
                out_keys=[(group, "obs_action")],
            )

            critic_module = TensorDictModule(
                module = MultiAgentMLP(
                    n_agent_inputs= self.env.observation_spec[group, "observation"].shape[-1]
                    + self.env.full_action_spec[group, "action"].shape[-1],
                    n_agent_outputs=1,
                    n_agents = len(agents),
                    centralized=centralized,
                    share_params= share_critic_params,
                    device = self.device,
                    depth = 2,
                    num_cells = 256,
                    activation_class= torch.nn.Tanh
                ),
                in_keys=[(group, "obs_action")],
                out_keys=[(group, "state_action_value")],
            )

            critics[group] = TensorDictSequential(
                cat_module,
                critic_module
            )
        return critics


    def _resume_modules(self):
        modules = {}
        for group in self.env.group_map.keys():
            modules[f"loss_{group}"] = self.losses[group]
            modules[f"policy_{group}"] = self.policies[group]
            modules[f"critic_{group}"] = self.critics[group]
            modules[f"opt_actor_{group}"] = self.optimisers[group]["loss_actor"]
            modules[f"opt_value_{group}"] = self.optimisers[group]["loss_value"]
            modules[f"noise_{group}"] = self.exploration_policies[group][-1]
        return modules

    def _resume_buffers(self):
        return {f"replay_{g}": b for g, b in self.replay_buffers.items()}

    def train(self):
        print("Training MADDPG Experiment...")

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

        for offset, batch in enumerate(self.collector):
            iteration = start_iteration + offset
            current_frames = batch.numel()
            total_frames += current_frames
            batch = self.process_batch(batch)

            actor_losses = {group: [] for group in self.env.group_map.keys()}
            critic_losses = {group: [] for group in self.env.group_map.keys()}

            for group in train_group_map.keys():
                group_batch = batch.exclude(
                    *[
                        key 
                        for _group in self.env.group_map.keys()
                        if _group != group
                        for key in [_group, ("next", _group)]
                    ]
                )
                group_batch = group_batch.reshape(-1)
                self.replay_buffers[group].extend(group_batch)

                for _ in range(self.config.get('training').get('n_optimiser_steps')):
                    minibatch = self.replay_buffers[group].sample()

                    loss_vals = self.losses[group](minibatch)

                    actor_losses[group].append(loss_vals["loss_actor"].item())
                    critic_losses[group].append(loss_vals["loss_value"].item())

                    for loss_name in ["loss_actor", "loss_value"]:
                        loss = loss_vals[loss_name]
                        optimiser = self.optimisers[group][loss_name]
                        loss.backward()
                        params = optimiser.param_groups[0]['params']
                        torch.nn.utils.clip_grad_norm_(params, self.config.get('training').get('max_grad_norm'))
                        optimiser.step()
                        optimiser.zero_grad()

                    self.target_updaters[group].step()
                    total_train_steps += 1

                self.exploration_policies[group][-1].step(current_frames)

            # --- Dedicated evaluation (thinned by eval_interval; None rows are
            # skipped by MetricsLogger.log and dropped by the analysis scripts) ---
            eval_means = None
            if self.should_evaluate(iteration):
                eval_means = self.evaluate_at_iteration(iteration, n_episodes=20)

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

                self.metrics_logger.log(
                    iteration=iteration,
                    group=group,
                    elapsed_time=round(elapsed, 2),
                    episode=total_episodes,
                    step=total_frames,
                    train_step=total_train_steps,
                    speed_fps=round(speed, 2),
                    episode_reward_mean=episode_reward_mean,
                    eval_reward_mean=eval_means[group] if eval_means is not None else None,
                    actor_loss=round(sum(actor_losses[group]) / n_opt, 6),
                    critic_loss=round(sum(critic_losses[group]) / n_opt, 6),
                    replay_size=len(self.replay_buffers[group]),
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
                        f"episode_reward_mean_{group} = "
                        f"{self.metrics_logger.get_values('episode_reward_mean', group)[-1]}"
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
        recent = self.metrics_logger.get_values("episode_reward_mean", first_group)[-10:]
        return (
            f"MADDPG training complete. Environment: {self.config['scenario_name']}, "
            f"Experiment Type: {self.experiment_type}, Seed: {self.seed}.\n\n"
            f"Results: {recent}"
        )


    def _setup_data_collection(self):
        agents_exploration_policy = TensorDictSequential(*self.exploration_policies.values())

        collector = SyncDataCollector(
            self.env,
            agents_exploration_policy,
            frames_per_batch=self.config.get('frames_per_batch'),
            device=self.device,
            total_frames=self.config.get('total_frames'),
            # Resume checkpoints are batch-boundary snapshots. Resetting here
            # makes the next batch reproducible from restored RNG state alone.
            reset_at_each_iter=True,
        )

        return agents_exploration_policy, collector

    def _setup_replay_buffer(self):
        replay_buffers = {}
        for group, _agents in self.env.group_map.items():
            replay_buffer = ReplayBuffer(
                storage = LazyMemmapStorage(self.config.get('memory_size')),
                sampler = RandomSampler(),
                batch_size = self.config.get('training').get('train_batch_size'),
            )

            if self.device.type != "cpu":
                replay_buffer.append_transform(lambda td: td.to(self.device))
            replay_buffers[group] = replay_buffer
        return replay_buffers
    
    def _setup_loss_functions(self):
        losses = {}

        for group, _agents in self.env.group_map.items():
            # delay_actor: the TD target's next action comes from a Polyak
            # target actor, as in the original DDPG/MADDPG and as IBMARL's
            # bootstrap already does (arbiter.py uses target_rl_policies).
            # torchrl's default is False (online actor), which buzzwire4's
            # baselines ran with: every MADDPG/RLfD/RFT seed that collapsed did
            # so with Q climbing past the binary schema's ceiling of 1.  RLfD
            # and RFT inherit this method, so the change covers all three.
            loss_module = DDPGLoss(
                actor_network = self.policies[group],
                value_network = self.critics[group],
                delay_value = True,
                delay_actor = True,
            )
            loss_module.set_keys(
                state_action_value = (group, "state_action_value"),
                reward= (group, "reward"),
                done = (group, "done"),
                terminated = (group, "terminated"),
            )
            loss_module.make_value_estimator(ValueEstimators.TD0, gamma= self.config.get('training').get('gamma'))

            losses[group] = loss_module
        
        target_updater = {
            group: SoftUpdate(loss, tau= self.config.get('training').get('polyak_tau')) for group, loss in losses.items()
        }

        optimisers = {
            group: {
                "loss_actor": torch.optim.Adam(
                    loss.actor_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
                ),
                "loss_value": torch.optim.Adam(
                    loss.value_network_params.flatten_keys().values(), lr = float(self.config.get('training').get('lr'))
                )
            }
            for group, loss in losses.items()
        }

        return losses, target_updater, optimisers
    
    def process_batch(self, batch: TensorDictBase) -> TensorDictBase:
        """
        If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
        `"terminated"` and `"done"`.
        This is needed to present them with the same shape as the reward to the loss.
        """
        for group in self.env.group_map.keys():
            keys = list(batch.keys(True, True))
            group_shape = batch.get_item_shape(group)
            nested_done_key = ("next", group, "done")
            nested_terminated_key = ("next", group, "terminated")
            if nested_done_key not in keys:
                batch.set(
                    nested_done_key,
                    batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
                )
            if nested_terminated_key not in keys:
                batch.set(
                    nested_terminated_key,
                    batch.get(("next", "terminated"))
                    .unsqueeze(-1)
                    .expand((*group_shape, 1)),
                )
        return batch
    
    def save_results(self):
        """Also record a policy video, matching IBMARL's behaviour."""
        super().save_results()
        try:
            self.render_policy()
        except Exception as e:
            print(f"Could not save policy video: {e}")

    def render_policy(self):
        if getattr(self, "_video_created", False):
            return
        self._video_created = True
        try:
            self._record_policy_video(self.policies, tag="policy")
        except Exception as e:
            print(f"Could not render policy: {e}")
