import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

import networks
import tools
import vae_utils


class WorldModelCustom(nn.Module):
    """
    Cleaner WorldModel with optional CNVAE bottleneck.

    Key points:
    - self.embed_size: encoder output size (CNVAE: flattened latents; else MultiEncoder outdim)
    - self.feat_size:  RSSM feature size = dim(concat(stoch,deter)), computed once from RSSM.initial
    - CNVAE: no seed for encoder; seed (RSSM feat) used for KL & decoding
    - CNVAE recon returns NHWC; we only reshape, never permute.
    """

    def __init__(self, obs_space, act_space, step, config):
        super().__init__()
        self._step = step
        self._use_amp = (config.precision == 16)
        self._config = config

        self.use_cnvae   = getattr(config, "use_cnvae", False)
        self.use_poisson = getattr(config, "use_poisson", False)

        # ---- Shapes from obs space
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        img_ch = shapes.get("image", (0, 0, 3))[2]

        # ---- Encoder (CNVAE or MultiEncoder)
        if self.use_cnvae:
            cnvae_cfg = dict(getattr(config, "cnvae_cfg", {}))
            cnvae_cfg["in_channels"] = img_ch

            # embed_size = #groups * latent_dim (compress=True)
            latent_dim = cnvae_cfg.get("latent_dim", 32)
            groups     = cnvae_cfg.get("groups", [2, 2, 2, 1])
            self.embed_size = latent_dim * sum(groups)

            # We'll fill cnvae_cfg["feat_size"] after RSSM is built (we need RSSM feat dim)
            self._cnvae_cfg = cnvae_cfg
            self._encoder = None  # unified interface below
        else:
            self._encoder = networks.MultiEncoder(shapes, **config.encoder)
            self.embed_size = self._encoder.outdim

        # ---- RSSM (single source of truth for feat_size)
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
            self.embed_size,      # encoder embed size into RSSM
            config.device,
            use_poisson=self.use_poisson,
            poisson_temp=getattr(config, "poisson_temp", 1.0),
        )

        if self.use_poisson:
            zdim = config.dyn_stoch
        elif config.dyn_discrete:
            zdim = config.dyn_stoch * config.dyn_discrete
        else:
            zdim = config.dyn_stoch
        self.feat_size = zdim + config.dyn_deter

        # If CNVAE, now we can instantiate it with known feat_size for seed projection
        if self.use_cnvae:
            self._cnvae_cfg["feat_size"] = self.feat_size
            self.bottleneck = networks.CNVAE(
                self._cnvae_cfg, use_poisson=self.use_poisson, use_cat=False
            )
            # Sanity: seed projection present
            assert getattr(self.bottleneck, "seed_proj", None) is not None, "CNVAE seed_proj missing"

        # ---- Heads
        self.heads = nn.ModuleDict()

        if not self.use_cnvae:
            # Vanilla Dreamer image decoder
            self.heads["decoder"] = networks.MultiDecoder(
                self.feat_size, shapes, **config.decoder
            )

        # Reward & continuation always from RSSM feat
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
            assert name in self.heads or (name == "decoder" and self.use_cnvae), name

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
        print(f"Optimizer model_opt has {sum(p.numel() for p in self.parameters())} variables.")

        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

        self.to(config.device)

    # ---------------------------------------------------------------------
    # Unified encode API
    # ---------------------------------------------------------------------
    @property
    def encoder(self):
        return self._encode_obs

    def _encode_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return [B,T,E] or [B,E] like MultiEncoder."""
        img = obs["image"]   # [B,T,H,W,C] or [B,H,W,C]
        if not self.use_cnvae:
            return self._encoder(obs)

        # CNVAE: produce flattened latents per frame, no seed here
        if img.dim() == 5:
            B, T, H, W, C = img.shape
            x = img.view(B * T, H, W, C)  # CNVAE handles NHWC internally
            latents, _, _, _, _ = self.bottleneck.xtract_ftr(x, full=False, seed=None)
            zflat = torch.cat([z.reshape(B * T, -1) for z in latents], dim=1)
            return zflat.view(B, T, -1)
        elif img.dim() == 4:
            B, H, W, C = img.shape
            latents, _, _, _, _ = self.bottleneck.xtract_ftr(img, full=False, seed=None)
            zflat = torch.cat([z.reshape(B, -1) for z in latents], dim=1)
            return zflat
        else:
            raise ValueError(f"Unexpected image shape {tuple(img.shape)}")

    # ---------------------------------------------------------------------
    # Preprocess
    # ---------------------------------------------------------------------
    def preprocess(self, obs):
        obs = {k: torch.tensor(v, device=self._config.device, dtype=torch.float32) for k, v in obs.items()}
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            obs["discount"] = (obs["discount"] * self._config.discount).unsqueeze(-1)
        assert "is_first" in obs and "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def _train(self, data):
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                # 1) Encode
                embed = self.encoder(data)  # [B,T,E] or [B,E]

                # 2) RSSM observe
                post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])

                # 3) KL terms
                total_kl_loss = 0.0
                total_kl_value = 0.0

                # 3.a) CNVAE KL (seed with prior features)
                if self.use_cnvae:
                    feat_prior = self.dynamics.get_feat(prior).reshape(-1, self.feat_size)  # [B*T, F]
                    img_bt = data["image"].reshape(-1, *data["image"].shape[2:])            # [B*T, H, W, C]
                    _, _, _, q_all, p_all = self.bottleneck.xtract_ftr(img_bt, full=False, seed=feat_prior)

                    # Balance or simple sum
                    kl_levels, _ = self.bottleneck.loss_kl(q_all, p_all)  # list of [B*T]
                    if hasattr(self, "_alphas") and hasattr(self, "_betas") and self._step < len(self._betas):
                        beta = float(self._betas[self._step])
                        kl_term, _, _ = vae_utils.kl_balancer(
                            kl_levels,
                            alpha=self._alphas,
                            coeff=beta,
                            beta=getattr(self._config, "kl_beta", 1.0),
                        )
                        cnvae_kl_loss = kl_term
                    else:
                        cnvae_kl_loss = sum(kl_levels) / len(kl_levels)

                    total_kl_loss = total_kl_loss + cnvae_kl_loss
                    total_kl_value = total_kl_value + cnvae_kl_loss.detach()

                # 3.b) RSSM KL
                rssm_kl_loss, rssm_kl_value, dyn_loss, rep_loss, kl_info = self._rssm_kl(post, prior)
                total_kl_loss = total_kl_loss + rssm_kl_loss
                total_kl_value = total_kl_value + rssm_kl_value

                # 4) Heads (reward/cont) from post features
                preds: Dict[str, Any] = {}
                feat_post = self.dynamics.get_feat(post)
                for name, head in self.heads.items():
                    if name == "decoder" and self.use_cnvae:
                        continue
                    grad_head = name in self._config.grad_heads
                    feat = feat_post if grad_head else feat_post.detach()
                    pred = head(feat)
                    preds.update(pred if isinstance(pred, dict) else {name: pred})

                # 5) Image reconstruction
                if self.use_cnvae:
                    # Reconstruct with prior seed (teacher-forced frames)
                    feat_prior = self.dynamics.get_feat(prior).reshape(-1, self.feat_size)
                    img_bt = data["image"].reshape(-1, *data["image"].shape[2:])  # [B*T,H,W,C]
                    _, _, recon_bt, _, _ = self.bottleneck.xtract_ftr(img_bt, full=True, seed=feat_prior)
                    B, T = data["image"].shape[:2]
                    recon = recon_bt.view(B, T, *data["image"].shape[2:])  # NHWC
                    # Fixed-variance normal NLL (like decoder="normal")
                    recon_loss = -torch.distributions.Normal(recon, 1.0).log_prob(data["image"]).sum(dim=[2, 3, 4])
                else:
                    # Vanilla Dreamer decoder distribution
                    img_dist = self.heads["decoder"](feat_post)["image"]
                    preds["image"] = img_dist

                # 6) Losses
                losses = {}
                for name, pred in preds.items():
                    if name == "image" and self.use_cnvae:
                        losses[name] = recon_loss
                    else:
                        losses[name] = -pred.log_prob(data[name])
                    assert losses[name].shape == data["image"].shape[:2], (name, losses[name].shape)

                scaled = {k: v * self._scales.get(k, 1.0) for k, v in losses.items()}
                model_loss = sum(scaled.values()) + total_kl_loss

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        # Metrics
        metrics.update({f"{k}_loss": tools.to_np(v) for k, v in losses.items()})
        metrics["dyn_loss"] = tools.to_np(dyn_loss)
        metrics["rep_loss"] = tools.to_np(rep_loss)
        metrics["kl"] = tools.to_np(torch.mean(total_kl_value))
        if kl_info:
            for key, value in kl_info.items():
                if torch.is_tensor(value):
                    metrics[f"kl_{key}"] = tools.to_np(torch.mean(value))

        # Context
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=total_kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )

        post = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in post.items()}
        return post, context, metrics

    # ---------------------------------------------------------------------
    # Video prediction
    # ---------------------------------------------------------------------
    def video_pred(self, data):
        data = self.preprocess(data)

        if self.use_cnvae:
            # Encode observed frames, update RSSM on first 5 steps
            embed = self.encoder(data)
            states, _ = self.dynamics.observe(
                embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
            )

            # Reconstruct observed with seed from states
            feat_obs = self.dynamics.get_feat(states).reshape(-1, self.feat_size)
            obs_img_bt = data["image"][:6, :5].reshape(-1, *data["image"].shape[2:])
            _, _, recon_obs_bt, _, _ = self.bottleneck.xtract_ftr(obs_img_bt, full=True, seed=feat_obs)
            recon = recon_obs_bt.view(6, 5, *data["image"].shape[2:])  # NHWC

            # Imagine future
            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
            feat_imag = self.dynamics.get_feat(prior).reshape(-1, self.feat_size)

            T_imag = data["action"][:6, 5:].shape[1]
            obs_like_bt = torch.zeros(6 * T_imag, *data["image"].shape[2:], device=data["image"].device)
            _, _, openl_bt, _, _ = self.bottleneck.xtract_ftr(obs_like_bt, full=True, seed=feat_imag)
            openl = openl_bt.view(6, T_imag, *data["image"].shape[2:])  # NHWC
        else:
            embed = self._encoder(data)
            states, _ = self.dynamics.observe(
                embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
            )
            recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[:6]
            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
            openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()

        # Reward predictions
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()

        # Compose video: truth | recon | error
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = torch.clamp(model, 0.0, 1.0)
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _rssm_kl(self, post, prior):
        cfg = self._config
        return self.dynamics.kl_loss(
            post, prior,
            free=cfg.kl_free,
            dyn_scale=cfg.dyn_scale,
            rep_scale=cfg.rep_scale,
            step=self._step,
            free_bits_max=getattr(cfg, "kl_free_bits_max", 1.0),
            anneal_steps=getattr(cfg, "kl_anneal_steps", 50000),
        )