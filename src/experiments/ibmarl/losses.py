'''
handles losses and optimization
'''
import torch
import torch.nn.functional as F

from tensordict import TensorDict, TensorDictBase



class GroupTrainer:
    def __init__(self, cfg, policies, critics, target_policies, target_critics, action_arbiter, env):

        #parameters
        self.lr = float(cfg["training"]["lr"])
        self.tau = float(cfg["training"]["polyak_tau"])
        self.max_grad_norm = float(cfg["training"]["max_grad_norm"])
        self.gamma =  float(cfg["training"]["gamma"])


        self.env = env
        self.action_arbiter = action_arbiter


        #networks
        self.rl_policies = policies
        self.critics = critics
        self.target_rl_policies = target_policies
        self.target_critics = target_critics

        self.num_critics = cfg["num_critics"]

        # Gated imitation term (CONTEXT.md).  Off unless actor_reg.mode is set,
        # in which case the plain MADDPG actor loss is unchanged and a term is
        # *added* -- see _ibmarl_actor_loss.
        reg = cfg.get("actor_reg") or {}
        self.reg_mode = str(reg.get("mode", "none"))
        if self.reg_mode not in ("none", "gated", "uniform"):
            raise ValueError(
                f"actor_reg.mode must be none|gated|uniform, got {self.reg_mode!r}"
            )
        self.reg_alpha = float(reg.get("alpha", 0.0))
        self.reg_hard_gate = str(reg.get("gate", "soft")) == "hard"
        temp = reg.get("temperature")
        self.reg_temperature = None if temp is None else float(temp)

        self.optimisers = self._build_optimisers()




    def update_critic(self, group, minibatch) -> dict:
        loss_value = self._ibmarl_value_loss(group, minibatch)
        critic_loss_val = loss_value.item()

        opt_critic = self.optimisers[group]["loss_value"]
        opt_critic.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(opt_critic.param_groups[0]["params"], self.max_grad_norm)
        opt_critic.step()

        return {"critic_loss": critic_loss_val}

    def update_actor(self, group, minibatch) -> dict:
        loss_actor, reg_info = self._ibmarl_actor_loss(group, minibatch)
        actor_loss_val = loss_actor.item()

        opt_actor = self.optimisers[group]["loss_actor"]
        opt_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(opt_actor.param_groups[0]["params"], self.max_grad_norm)
        opt_actor.step()

        return {"actor_loss": actor_loss_val, **reg_info}
    
    
    def polyak_step(self, source, target):
        '''
        updates target networks which handles the critic ensemble and the actor
        '''
        if isinstance(source, torch.nn.ModuleList):
            #iterate through the ensembles:
            for s_mod, t_mod in zip(source,target):
                self._single_polyak(s_mod, t_mod)
        else:
            self._single_polyak(source, target)


    def _single_polyak(self,source:torch.nn.Module, target: torch.nn.Module):
        '''
        updates target networks
        '''
        for p, p_targ in zip(source.parameters(), target.parameters()):
            p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
    
    def _build_optimisers(self):
        # Every ensemble member is trained. Their parameters go into a single
        # flat param group so the grad-clip in update_critic (which reads
        # param_groups[0]) still covers the whole ensemble.
        optimisers = {}
        for group in self.env.group_map.keys():
            critic_params = [
                p for critic in self.critics[group] for p in critic.parameters()
            ]
            optimisers[group] = {
                "loss_actor": torch.optim.Adam(
                    self.rl_policies[group].parameters(),
                    lr=self.lr,
                ),
                "loss_value": torch.optim.Adam(
                    critic_params,
                    lr=self.lr,
                ),
            }
        return optimisers


    def _ibmarl_value_loss(self, group: str, mb: TensorDictBase) -> torch.Tensor:
        """
        TD0 value loss with bootstrap proposal (IL vs RL) for next action.
        Every ensemble member regresses against the same target y; members differ
        only by initialisation, which is what makes the arbiter's min over a random
        pair a real pessimistic estimate rather than noise. Uses terminated for the
        bootstrap mask (no bootstrap on episode end), same as TorchRL convention.
        """
        obs = mb[(group, "observation")]
        act = mb[(group, "action")]
        rew = mb[("next", group, "reward")]
        next_obs = mb[("next", group, "observation")]

        # Use terminated for bootstrap mask (same as MADDPG/DDPGLoss)
        terminated = mb.get(("next", group, "terminated"), mb.get(("next", group, "done")))
        if terminated.dim() == 2:
            terminated = terminated.unsqueeze(-1)
        if terminated.shape[-2] != obs.shape[1]:
            terminated = terminated.expand(terminated.shape[0], obs.shape[1], 1)

        with torch.no_grad():
            _, q_next_val = self.action_arbiter.bootstrap_proposal(group, next_obs)
            q_target_val = q_next_val.unsqueeze(-1)

            if rew.dim() < 3:
                rew = rew.unsqueeze(-1)

            # TD0: y = r + gamma * (1 - terminated) * Q(s', a'_star)
            y = rew + self.gamma * (1.0 - terminated.float()) * q_target_val

        td_cur = TensorDict(
            {(group, "observation"): obs, (group, "action"): act},
            batch_size=[obs.shape[0]],
            device=obs.device,
        )
        # A fresh td per member: the critic TensorDictModule writes obs_action and
        # state_action_value into whatever td it is handed, so reusing one across
        # members would overwrite intermediates.
        member_losses = [
            F.mse_loss(critic(td_cur.clone())[(group, "state_action_value")], y)
            for critic in self.critics[group]
        ]
        # Mean, not sum, so critic_loss stays on the same scale as the previous
        # single-critic number and remains comparable across runs.
        return torch.stack(member_losses).mean()
    
    def _ibmarl_actor_loss(self, group: str, mb: TensorDictBase):
        """
        MADDPG-style actor loss (same as RLFD): L_actor = -E[ Q(obs, pi(obs)) ].
        Deliberately scores against critic 0 alone rather than the ensemble: members
        are trained on identical targets from identical minibatches, so the choice is
        near-neutral, and keeping a single critic preserves parity with the
        MADDPG/RLFD baselines this method is compared against.

        With ``actor_reg.mode`` set, the gated imitation term is added:

            alpha * mean|Q_tot| * E_o[ gate(o) * ||pi_theta(o) - pi_teacher(o)||^2 ]

        The DPG term above can only climb Q locally at the actor's own action, so
        it never learns from the teacher's executed actions even when the critic
        scores them higher (the actor lags the arbitrated policy).  The added term
        pulls the actor straight toward the teacher's proposal, weighted by the
        arbiter's own preference for it at that observation, so it switches off
        state by state as the actor overtakes the teacher.  ``uniform`` holds the
        gate open (the RFT-like control).  The term is scaled by the detached
        minibatch mean |Q_tot| rather than the DPG term being normalised, so the
        ``none`` path and its grad-clip behaviour are byte-identical to before.

        Returns ``(loss, info)`` where info carries the diagnostics ``reg_mse``
        (unweighted mean squared distance to the teacher) and ``gate_mean``
        (mean gate over the minibatch), both None when the term is off.
        """
        obs = mb[(group, "observation")]
        B = obs.shape[0]

        td_pi = TensorDict(
            {(group, "observation"): obs}, batch_size=[B], device=obs.device
        )
        td_pi = self.rl_policies[group](td_pi)
        a_pi = td_pi[(group, "action")]

        td_q = TensorDict(
            {(group, "observation"): obs, (group, "action"): a_pi},
            batch_size=[B],
            device=obs.device,
        )
        q = self.critics[group][0](td_q)[(group, "state_action_value")]

        if q.dim() == 3:
            q_tot = q.sum(dim=1).squeeze(-1)
        else:
            q_tot = q.squeeze(-1)
        loss = -q_tot.mean()
        info = {"reg_mse": None, "gate_mean": None}
        if self.reg_mode == "none":
            return loss, info

        with torch.no_grad():
            a_il = self.action_arbiter.il_policy.get_action(group, obs)   # [B, N, A]
            if self.reg_mode == "gated":
                gate, _, _ = self.action_arbiter.teacher_preference(
                    group,
                    obs,
                    a_pi.detach(),
                    temperature=self.reg_temperature,
                    hard=self.reg_hard_gate,
                )
            else:
                gate = torch.ones(B, device=obs.device)
            q_scale = q_tot.abs().mean()
        bc = ((a_pi - a_il) ** 2).mean(dim=(1, 2))                          # [B]
        loss = loss + self.reg_alpha * q_scale * (gate * bc).mean()
        info = {"reg_mse": bc.mean().item(), "gate_mean": gate.mean().item()}
        return loss, info
