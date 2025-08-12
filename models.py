import copy
import torch
from torch import nn
import torch.nn.functional as F

import networks
import tools


to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # Mutate the buffer outside autograd to avoid inplace operation conflicts
        with torch.no_grad():
            ema_vals.lerp_(x_quantile, self.alpha)
            # ema_vals.copy_(self.alpha * x_quantile + (1 - self.alpha) * ema_vals)
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


#-----------------------------------------------------------------
# Original WorldModel
# ----------------------------------------------------------------
class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision != 32 else False
        self._amp_dtype = torch.bfloat16 if config.precision == 'bfloat16' else torch.float16

        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            self.feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            self.feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            self.feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()

        self._use_amp = True if config.precision != 32 else False
        self._amp_dtype = torch.bfloat16 if config.precision == 'bfloat16' else torch.float16

        self._config = config
        self._world_model = world_model

        # ask the world mdoel for the feature size (it should know how much it is exposing to the actor / value)
        feat_size = world_model.feat_size

        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            # Seed EMA range to a small, sane window to avoid early exploding advantages.
            init_ema = torch.tensor([-1.0, 1.0], device=self._config.device)
            self.register_buffer("ema_vals", init_ema.clone())
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):

        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                dyn = (self._config.imag_gradient in ["dynamics", "both"])
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, allow_grad=dyn
                )
                if not dyn:
                    imag_feat = imag_feat.detach()
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                # state_ent = self._world_model.dynamics.get_dist(imag_state).entropy() # unused values
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                # actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = actor_loss - self._config.actor["entropy"] * actor_ent[:-1, ..., None]

                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                if self._config.critic["slow_target"]:
                    # Compute slow critic distribution without tracking gradients
                    with torch.no_grad():
                        slow_v = self._slow_value(value_input[:-1].detach())
                    # Regularize online critic toward slow critic in distribution space
                    value_loss = value_loss - slow_v.log_prob(value.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            # SAFE POLYAK UPDATE: must run after all backward/optimizer steps
            if hasattr(self, "_polyak_update"):
                self._polyak_update()
        return imag_feat, imag_state, imag_action, weights, metrics


    def _imagine(self, start, policy, horizon, allow_grad=True):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))

        # handle both hierarchical and flat latents
        start = {
            k: [flatten(x) for x in v] if isinstance(v, list) else flatten(v)
            for k, v in start.items()
        }

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            action = policy(feat).sample()
            succ = dynamics.img_step(state, action)
            if isinstance(succ, tuple):
                succ = succ[0] # hRSSM returns a tuple of (prior, spatial)

            if allow_grad:
                next_state = succ
            else:
                next_state = {k: [item.detach() for item in v] if isinstance(v, list) else v.detach() for k, v in succ.items()}

            return next_state, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        # states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()} # old flat RSSM only version
        states = {}
        for k, v_seq in succ.items():
            if isinstance(v_seq, list):
                # Hierarchical case for keys like 'stoch', 'logit', etc.
                # start[k] is a list of tensors for each level.
                # v_seq is also a list of tensors for each level.
                states[k] = [
                    torch.cat([start_tensor[None], seq_tensor[:-1]], 0)
                    for start_tensor, seq_tensor in zip(start[k], v_seq)
                ]
            else:
                # Flat case for 'deter'.
                states[k] = torch.cat([start[k][None], v_seq[:-1]], 0)

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        dyn = (self._config.imag_gradient in ["dynamics", "both"])
        inp = imag_feat if dyn else imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    # def _update_slow_target(self): # potential in place bug
    #     if self._config.critic["slow_target"]:
    #         if self._updates % self._config.critic["slow_target_update"] == 0:
    #             mix = self._config.critic["slow_target_fraction"]
    #             for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
    #                 d.data = mix * s.data + (1 - mix) * d.data
    #         self._updates += 1
    # ------------------------------------------------------------------------ #
    #  SAFE POLYAK UPDATE (runs after backprop)                                #
    # ------------------------------------------------------------------------ #
    def _polyak_update(self):
        if not self._config.critic.get("slow_target", True):
            return
        # update every K steps
        K = self._config.critic.get("slow_target_update", 1)
        if (self._updates % K) != 0:
            self._updates += 1
            return
        tau = float(self._config.critic.get("slow_target_fraction", 0.005))
        with torch.no_grad():
            for p_src, p_tgt in zip(self.value.parameters(), self._slow_value.parameters()):
                p_tgt.lerp_(p_src, tau)
        self._updates += 1


#-----------------------------------------------------------------
# Hierarchical World Model
# ----------------------------------------------------------------

class DecoderHead(nn.Module):
    """
    Decoder that reconstructs the image from the final spatial feature map.
    Optionally applies FiLM conditioning from the top-level latent z at the
    coarsest scale to encourage the decoder to use the hierarchy.
    """
    def __init__(self, input_channels, output_shape=(3, 64, 64), cnn_sigmoid=False,
                 film_top_level=True, z_top_dim=None):
        super().__init__()
        self._output_shape = output_shape
        self._cnn_sigmoid = cnn_sigmoid
        self._film_top_level = bool(film_top_level and (z_top_dim is not None))
        # FiLM MLPs map z_top -> per-channel scale/shift for spatial features
        if self._film_top_level:
            self.film_gamma = nn.Linear(int(z_top_dim), int(input_channels))
            self.film_beta  = nn.Linear(int(z_top_dim), int(input_channels))
        # Final 1x1 convolution to image channels
        self.conv = nn.Conv2d(input_channels, output_shape[0], kernel_size=1)
        print(f"DecoderHead initialized to output shape {output_shape}.")
        print(f"Input channels: {input_channels}, FiLM: {self._film_top_level}")

    def forward(self, spatial_features, z_top=None):
        # Accept (B,T,C,H,W) or (BT,C,H,W)
        if spatial_features.ndim == 5:
            B, T, C, H, W = spatial_features.shape
            x = spatial_features.reshape(B*T, C, H, W)
            batch_shape = (B, T)
        else:
            BT, C, H, W = spatial_features.shape
            x = spatial_features
            batch_shape = (BT,)

        target_hw = self._output_shape[1], self._output_shape[2]
        if (H, W) != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)

        # Optional FiLM using top-level latent
        if self._film_top_level and z_top is not None:
            # z_top: (B,T,Dim) or (BT,Dim)
            if isinstance(z_top, (list, tuple)):
                z_top = z_top[-1]
            if z_top.ndim == 3:
                zt = z_top.reshape(-1, z_top.shape[-1])
            else:
                zt = z_top
            gamma = self.film_gamma(zt).view(x.shape[0], x.shape[1], 1, 1)
            beta  = self.film_beta(zt).view(x.shape[0], x.shape[1], 1, 1)
            x = gamma * x + beta

        mean = self.conv(x)  # (BT, C_out, Ht, Wt) with (Ht,Wt)==target_hw
        if len(batch_shape) == 2:
            B, T = batch_shape
            mean = mean.reshape(B, T, *self._output_shape)
        else:
            mean = mean.reshape(batch_shape[0], 1, *self._output_shape)  # degenerate T=1

        # Convert to channel-last for image loss: (B,T,C,H,W) -> (B,T,H,W,C)
        mean = mean.permute(0, 1, 3, 4, 2)
        mean = torch.sigmoid(mean) if self._cnn_sigmoid else (mean + 0.5)
        return {'image': tools.MSEDist(mean)}


class LadderDecoder(nn.Module):
    """
    CNVAE-style ladder decoder:
    - Start from the coarsest latent z_{L-1} expanded to spatial.
    - Iteratively upsample and inject lower-level latents z_{l} via concat+1x1 merge.
    - A few 3x3 conv blocks at each level provide spatial mixing.
    - Produces final RGB at the finest resolution.

    Inputs at runtime: list of latents per level [z_l] with order finest->coarsest,
    shaped (B,T,S,K) for discrete or (B,T,S) for continuous. We internally flatten.
    """
    def __init__(self,
                 z_flat_ch,           # list[int], channels after flatten per level (finest->coarsest)
                 spatial_sizes,       # list[int], spatial size per level (finest->coarsest)
                 out_shape=(3, 64, 64),
                 blocks_per_level=2,
                 up_mode='bilinear',
                 cnn_sigmoid=False):
        super().__init__()
        assert len(z_flat_ch) == len(spatial_sizes)
        self.L = len(z_flat_ch)
        self.z_flat_ch = list(z_flat_ch)
        self.spatial_sizes = list(spatial_sizes)
        self.out_shape = out_shape
        self.blocks_per_level = int(blocks_per_level)
        self.up_mode = up_mode
        self.cnn_sigmoid = cnn_sigmoid

        # We build ExpandZ to map flat z_l -> (Cz_l, H_l, H_l);
        # match hRSSM choice: out_ch = z_flat_ch[l].
        self.expand_z = nn.ModuleList([
            networks.ExpandZ(in_ch=self.z_flat_ch[l], out_ch=self.z_flat_ch[l], spatial=self.spatial_sizes[l])
            for l in range(self.L)
        ])

        # Merge 1x1 to bring concat(prev_feat, z_l_feat) back to z_flat_ch[l].
        self.merge = nn.ModuleList()
        self.level_blocks = nn.ModuleList()
        for l in range(self.L):
            ch = self.z_flat_ch[l]
            if l == self.L - 1:
                in_ch = ch
            else:
                in_ch = ch + self.z_flat_ch[l + 1]
            self.merge.append(nn.Conv2d(in_ch, ch, kernel_size=1))
            # simple residual-style stack
            blocks = []
            for _ in range(self.blocks_per_level):
                blocks += [nn.Conv2d(ch, ch, kernel_size=3, padding=1), nn.SiLU()]
            self.level_blocks.append(nn.Sequential(*blocks))

        # Final refinement and 1x1 to image channels.
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.z_flat_ch[0], self.z_flat_ch[0], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.z_flat_ch[0], out_shape[0], kernel_size=1),
        )

    def _BT(self, x):
        # x: (B,T,...) -> (BT,...)
        assert x.ndim >= 3
        B, T = x.shape[:2]
        return B, T, x.reshape(B * T, *x.shape[2:])

    def _interpolate(self, x, size):
        if 'linear' in self.up_mode:
            return F.interpolate(x, size=(size, size), mode=self.up_mode, align_corners=False)
        return F.interpolate(x, size=(size, size), mode=self.up_mode)

    def forward(self, z_list):
        # z_list expected finest->coarsest
        assert isinstance(z_list, (list, tuple)) and len(z_list) == self.L
        # Extract shapes
        B, T = z_list[0].shape[0], z_list[0].shape[1]
        # Work with BT batch
        def flat_bt(z):
            z = z.reshape(B * T, -1)
            return z

        # Start from coarsest
        l = self.L - 1
        z_top = flat_bt(z_list[l])
        feat = self.expand_z[l](z_top)               # (BT, Cz_l, H_l, H_l)
        feat = self.merge[l](feat)
        feat = self.level_blocks[l](feat)

        # Progressively add finer levels
        for l in range(self.L - 2, -1, -1):
            H = int(self.spatial_sizes[l])
            if feat.shape[-1] != H:
                feat = self._interpolate(feat, H)
            z_flat = flat_bt(z_list[l])
            z_spat = self.expand_z[l](z_flat)
            feat = torch.cat([feat, z_spat], dim=1)
            feat = self.merge[l](feat)
            feat = self.level_blocks[l](feat)

        # Upsample to target image resolution if needed, then produce image and reshape
        C, H, W = self.out_shape
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        mean = self.final_conv(feat)                  # (BT, C, H, W)
        mean = mean.reshape(B, T, C, H, W)
        mean = mean.permute(0, 1, 3, 4, 2)           # (B,T,H,W,C)
        mean = torch.sigmoid(mean) if self.cnn_sigmoid else (mean + 0.5)
        return {'image': tools.MSEDist(mean)}

class hWorldModel(WorldModel):
    """
    Hierarchical World Model.
    Orchestrates the hConvEncoder, hRSSM, and a simplified decoder head.
    """
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision != 32 else False
        self._amp_dtype = torch.bfloat16 if config.precision == 'bfloat16' else torch.float16

        self._config = config

        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        self.encoder = networks.hConvEncoder(shapes['image'], **config.encoder)

        self.dynamics = networks.hRSSM(
            h_encoder_dims=list(self.encoder.out_channels),
            **config.rssm,
            device=config.device,
            num_actions=config.num_actions,
            unimix_ratio=config.unimix_ratio,
            initial=config.initial,
            dyn_mean_act=config.dyn_mean_act,
            dyn_std_act=config.dyn_std_act,
            dyn_min_std=config.dyn_min_std,
        )

        self.heads = nn.ModuleDict()

        # Use LadderDecoder: inject per-level latents in a top-down path
        H, W, C = shapes['image']
        self.heads["decoder"] = LadderDecoder(
            z_flat_ch=self.dynamics._z_flat_ch,              # finest->coarsest
            spatial_sizes=self.dynamics._spatial_sizes,      # finest->coarsest
            out_shape=(C, H, W),
            blocks_per_level=getattr(config, 'ladder_blocks_per_level', 2),
            up_mode=self.dynamics._up_mode if hasattr(self.dynamics, '_up_mode') else 'bilinear',
            cnn_sigmoid=config.decoder['cnn_sigmoid'],
        )

        self.feat_size = self.dynamics.feat_size() #sum(config.rssm['h_stoch_dims']) + sum(config.rssm['h_deter_dims'])

        self.heads["reward"] = networks.MLP(
            self.feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            self.feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )

        for name in config.grad_heads:
            assert name in self.heads, name

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer h_model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast('cuda', enabled=self._use_amp):
                embed_list = self.encoder(data)

                # hRSSM.observe now returns the final spatial feature map as the second element
                post, prior, spatial_post = self.dynamics.observe(
                    embed_list, data["action"], data["is_first"]
                )

                # Compute capacity schedule (free-bits warmup)
                if hasattr(self._config, "kl_anneal_steps"):
                    step_val = float(self._step()) if callable(self._step) else float(self._step)
                    progress = min(1.0, step_val / max(1.0, float(self._config.kl_anneal_steps)))
                    # Start from base free (kl_free) and anneal up to kl_free_bits_max
                    base_free = float(self._config.kl_free)
                    free_max = float(getattr(self._config, "kl_free_bits_max", base_free))
                    kl_free_total = base_free + (free_max - base_free) * progress
                else:
                    kl_free_total = float(self._config.kl_free)

                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free_total, self._config.dyn_scale, self._config.rep_scale
                )

                preds = {}
                # Ladder decoder consumes per-level latents (finest->coarsest)
                decoder_pred = self.heads['decoder'](post['stoch'])
                preds.update(decoder_pred)

                for name, head in self.heads.items():
                    if name == 'decoder': continue
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    preds[name] = pred

                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    losses[name] = loss

                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free_total
        metrics["dyn_scale"] = self._config.dyn_scale
        metrics["rep_scale"] = self._config.rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        # Optional per-level logs if available
        if hasattr(self.dynamics, "_kl_debug"):
            dbg = self.dynamics._kl_debug
            for i, v in enumerate(dbg["dyn_means"]):
                metrics[f"kl_dyn_l{i}"] = to_np(v)
            for i, v in enumerate(dbg["rep_means"]):
                metrics[f"kl_rep_l{i}"] = to_np(v)
            for i, v in enumerate(dbg["ema_dyn"]):
                metrics[f"kl_ema_dyn_l{i}"] = to_np(v)
            for i, v in enumerate(dbg["ema_rep"]):
                metrics[f"kl_ema_rep_l{i}"] = to_np(v)
            for i, v in enumerate(dbg["gamma_dyn"]):
                metrics[f"kl_gamma_dyn_l{i}"] = to_np(v)
            for i, v in enumerate(dbg["gamma_rep"]):
                metrics[f"kl_gamma_rep_l{i}"] = to_np(v)
        # Log what feature exposure the actor/value see as a numeric index for scalars
        level_cfg = None
        if hasattr(self._config, 'rssm'):
            level_cfg = self._config.rssm if isinstance(self._config.rssm, dict) else getattr(self._config.rssm, 'expose_levels', 'all')
        if isinstance(level_cfg, dict):
            level = level_cfg.get('expose_levels', 'all')
        else:
            level = level_cfg if level_cfg is not None else 'all'
        if level == 'all':
            idx = -1.0  # -1 means all levels
        elif level == 'top':
            idx = float(getattr(self.dynamics, '_h_levels', 1) - 1)
        else:
            try:
                idx = float(int(level))
            except Exception:
                idx = -1.0
        metrics["actor_feat_index"] = idx

        post_detached = {k: (v if not isinstance(v, list) else [i.detach() for i in v]) for k, v in post.items()}
        return post_detached, {}, metrics

    # # how many channels does _build_spatial() output?
    # def spatial_channels(self):
    #     z = sum(d * self._discrete if self._discrete else d
    #             for d in self._h_stoch_dims)
    #     d = sum(self._h_deter_dims)
    #     return z + d

    def preprocess(self, obs):
        obs = {
            k: (v.clone().detach().to(device=self._config.device, dtype=torch.float32)
                if isinstance(v, torch.Tensor)
                else torch.tensor(v, device=self._config.device, dtype=torch.float32))
            for k, v in obs.items()
        }
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = obs["discount"].unsqueeze(-1)
        assert "is_first" in obs
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs["image"] = obs["image"] / 255.0  # image normalized ot [0, 1]

        return obs

    def video_pred(self, data):
        data = self.preprocess(data)

        # Get initial state from a short context sequence
        context_len = 5
        embed_list = self.encoder(data)
        states, _, _ = self.dynamics.observe(
            [e[:, :context_len] for e in embed_list],
            data["action"][:, :context_len],
            data["is_first"][:, :context_len]
        )

        # Reconstructions for the context using ladder decoder (consume post latents)
        recon = self.heads["decoder"](states['stoch'])["image"].mode()

        # Get the last state to start imagination from
        init_state = {k: (v[:, -1] if not isinstance(v, list) else [vi[:, -1] for vi in v]) for k, v in states.items()}

        # Imagine the future
        priors, _ = self.dynamics.imagine_with_action(data["action"][:, context_len:], init_state)
        openl = self.heads["decoder"](priors['stoch'])["image"].mode()

        # Combine context reconstructions and open-loop predictions
        model = torch.cat([recon, openl], 1)
        truth = data["image"]
        error = (model - truth + 1.0) / 2.0 # Remap error to [0, 1] for visualization

        # Concatenate for side-by-side video comparison
        video = torch.cat([truth, model, error], 3)
        return video