# networks.py

import math
import itertools
import collections
import re
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools     # your project's helpers: weight_init, Cell, MULT, AddNorm, OneHotDist, ContDist, etc.


# ─────────────────────────────────────────────────────────────────────────────
#  Poisson latent distribution (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class Poisson:
    """Reparameterized Poisson with temperature smoothing."""
    def __init__(self, log_rate: torch.Tensor, t: float = 0.0):
        self.log_rate = log_rate.clamp(max=5.0)
        self.rate     = torch.exp(self.log_rate) + 1e-6
        self.n_trials = int(math.ceil(self.rate.max().item() * 5))
        self.t        = t

    def rsample(self, hard: bool=False):
        exp = torchd.Exponential(self.rate)
        x   = exp.rsample((self.n_trials,))   # [trials, *batch]
        times     = x.cumsum(0)
        indicator = (times < 1.0).float()
        if not hard and self.t>0:
            indicator = torch.sigmoid((1.0-times)/self.t)
        return indicator.sum(0)

    def sample(self):
        return self.rsample(hard=True)

    def log_prob(self, z):
        return z*self.log_rate - self.rate - torch.lgamma(z+1)

    def kl(self, prior, delta_log=None):
        if isinstance(prior, Poisson):
            logp, λp = prior.log_rate, prior.rate
        else:
            logp = prior.clamp(max=5.0)
            λp   = torch.exp(logp)+1e-6
        λq = self.rate
        δ  = (self.log_rate - logp) if delta_log is None else delta_log
        return λq - λp - λp*δ

    def entropy(self):
        return 0.5*torch.log(2*math.pi*math.e*self.rate)

    def mode(self):
        return torch.floor(self.rate)

    def mean(self):
        return self.rate


# ─────────────────────────────────────────────────────────────────────────────
#  Conv/DeConv same-pad, combiners, sampler
# ─────────────────────────────────────────────────────────────────────────────
class Conv2dSamePad(nn.Conv2d):
    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        pad_h = max((math.ceil(ih / sh) - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((math.ceil(iw / sw) - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

class DeConv2dSamePad(nn.ConvTranspose2d):
    def forward(self, x):
        return super().forward(x)

class CombinerEnc(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, 1)
    def forward(self, x1, x2):
        return x1 + self.conv(x2)

class CombinerDec(nn.Module):
    def __init__(self, ci1, ci2, co):
        super().__init__()
        self.conv = nn.Conv2d(ci1 + ci2, co, 1)
    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))

class Sampler(nn.Module):
    def __init__(self, in_ch, latent_chan, spatial_dim,
                 gaussian=True, compress=True, bias=True):
        super().__init__()
        out_ch = 2 * latent_chan if gaussian else latent_chan
        k = spatial_dim if compress else 1
        s = k if compress else 1
        self.conv = Conv2dSamePad(in_ch, out_ch, k, stride=s, padding=0, bias=bias)
    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Ladder-style CNVAE (conditioned by RSSM seed)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  CNVAE faithful to jcbyts/cnvae with optional RSSM seeding
# ─────────────────────────────────────────────────────────────────────────────
class CNVAE(nn.Module):
    """
    Ladder VAE:
      - Encoder tower with CombinerEnc taps
      - z0 from enc0(x); p(z0) standard Normal
      - For each subsequent group:
          p(z_i | s_dec) from dec_sampler on current decoder state
          q(z_i | enc, s_dec) from enc_sampler on CombinerEnc(enc_tap, s_dec)
          s_dec <- CombinerDec(s_dec, expand(z_i)) then local dec cells
      - Optional RSSM seeding: if cfg['feat_size'] is set and a seed is passed,
        the decoder is initialized from seed via a learned linear projection;
        otherwise a learned prior feature (like original cnvae) is used.
    """

    def __init__(self, cfg, use_poisson: bool = False, use_cat: bool = False):
        super().__init__()
        if use_poisson or use_cat:
            raise NotImplementedError("This CNVAE plug-in matches the original (Gaussian) CNVAE.")

        # ---- config
        self.scales      = cfg["scales"]          # e.g. [8, 16, 32, 64]
        self.groups      = cfg["groups"]          # per scale, e.g. [1,2,2,2]
        self.ch          = cfg["ch"]
        self.latent_dim  = cfg["latent_dim"]
        self.compress    = cfg.get("compress", True)
        self.act_fn_name = cfg.get("activation_fn", "SiLU")
        self.use_bn      = cfg.get("use_bn", True)
        self.use_se      = cfg.get("use_se", False)
        self.input_sz    = cfg["input_sz"]        # final H=W target
        self.in_channels = cfg["in_channels"]
        self.out_channels= cfg["out_channels"]
        self.ker_sz      = cfg["ker_sz"]
        self.n_pre_blocks= cfg["n_pre_blocks"]
        self.n_pre_cells = cfg["n_pre_cells"]
        self.n_enc_cells = cfg["n_enc_cells"]
        self.n_dec_cells = cfg["n_dec_cells"]
        self.n_post_blocks=cfg["n_post_blocks"]
        self.n_post_cells =cfg["n_post_cells"]
        self.feat_size   = cfg.get("feat_size", None)  # <- if provided, enables seeding

        Act = getattr(nn, self.act_fn_name)

        # ---- stem
        self.stem = Conv2dSamePad(self.in_channels, self.ch, self.ker_sz, padding=0)
        tools.weight_init(self.stem)

        # ---- pre-process (keep simple: normal_pre cells; do not downsample here)
        self.pre_process = nn.ModuleList()
        cell_cfg = {
            "n_nodes": cfg.get("n_nodes", 1),
            "act_fn": cfg.get("act_fn", self.act_fn_name),
            "use_bn": self.use_bn,
            "use_se": self.use_se,
            "scale":  cfg.get("scale", 1.0),
            "eps":    cfg.get("eps", 0.1),
        }
        for _b, _c in itertools.product(range(self.n_pre_blocks), range(self.n_pre_cells)):
            self.pre_process.append(tools.Cell(self.ch, self.ch, cell_type="normal_pre", **cell_cfg))

        # ---- encoder tower with Combiners
        enc, mult, depth = [], 1, 1
        for si, s in enumerate(self.scales):
            ch_in = int(self.ch * mult)
            for _ in range(self.n_enc_cells):
                enc.append(tools.Cell(ch_in, ch_in, cell_type="normal_enc", **cell_cfg))
                depth += 1
            # add one CombinerEnc per group at this scale, except the last group of the last scale
            for g in range(self.groups[si]):
                is_last_group_last_scale = (si == len(self.scales) - 1) and (g == self.groups[si] - 1)
                if not is_last_group_last_scale:
                    enc.append(CombinerEnc(ch_in, ch_in))
            # downsample between scales
            if si < len(self.scales) - 1:
                enc.append(tools.Cell(ch_in, ch_in * tools.MULT, cell_type="down_enc", **cell_cfg))
                mult *= tools.MULT
                depth += 1
        self.enc_tower = nn.ModuleList(enc)

        # ---- enc0 (before z0 sampler)
        final_ch = int(self.ch * mult)
        self.enc0 = nn.Sequential(Act(), Conv2dSamePad(final_ch, final_ch, 1, padding=0), Act())

        # ---- samplers & expanders (coarse -> fine order)
        self.enc_sampler = nn.ModuleList()
        self.dec_sampler = nn.ModuleList()
        self.expand      = nn.ModuleList()
        rev_scales  = list(reversed(self.scales))
        rev_groups  = list(reversed(self.groups))
        samp_mult   = float(mult)
        first_group = True
        for s, g in zip(rev_scales, rev_groups):
            ch_samp = int(self.ch * samp_mult)
            for _ in range(g):
                self.enc_sampler.append(Sampler(ch_samp, self.latent_dim, s, gaussian=True, compress=self.compress))
                if not first_group:
                    self.dec_sampler.append(Sampler(ch_samp, self.latent_dim, s, gaussian=True, compress=self.compress))
                first_group = False
                if self.compress:
                    self.expand.append(DeConv2dSamePad(self.latent_dim, self.latent_dim, s, stride=s, padding=0))
                else:
                    self.expand.append(nn.Identity())
            samp_mult /= float(tools.MULT)

        # ---- decoder tower (coarse -> fine)
        dec = nn.ModuleList()
        dec_mult = float(mult)
        for si, (s, g) in enumerate(zip(rev_scales, rev_groups)):
            ch_dec = int(self.ch * dec_mult)
            for gi in range(g):
                # like original: for the very first group at coarsest scale, skip local dec cells before first combiner
                if not (si == 0 and gi == 0):
                    for _ in range(self.n_dec_cells):
                        dec.append(tools.Cell(ch_dec, ch_dec, cell_type="normal_dec", **cell_cfg))
                dec.append(CombinerDec(ch_dec, self.latent_dim, ch_dec))
            if si < len(rev_scales) - 1:
                dec.append(tools.Cell(ch_dec, max(1, ch_dec // tools.MULT), cell_type="up_dec", **cell_cfg))
                dec_mult /= float(tools.MULT)
        self.dec_tower = dec

        # ---- post-process (finest)
        post_mult = float(dec_mult)
        post = []
        for b in range(self.n_post_blocks):
            for c in range(self.n_post_cells):
                ch_post = int(self.ch * post_mult)
                if c == 0:
                    post.append(tools.Cell(ch_post, max(1, ch_post // tools.MULT), cell_type="up_post", **cell_cfg))
                    post_mult /= float(tools.MULT)
                else:
                    post.append(tools.Cell(ch_post, ch_post, cell_type="normal_post", **cell_cfg))
            if b == self.n_post_blocks - 1:
                post.append(nn.Upsample(size=self.input_sz, mode="nearest"))
        if not len(post):
            post.append(nn.Upsample(size=self.input_sz, mode="nearest"))
        self.post_process = nn.ModuleList(post)

        # ---- output conv (image channels)
        self.out = nn.Conv2d(max(1, int(self.ch * post_mult)), self.out_channels, 3, padding=1)
        tools.weight_init(self.out)

        # ---- learned prior feature for the coarsest decoder state (original behavior)
        self.coarsest = rev_scales[0]
        self.dec_ch0  = int(self.ch * float(mult))
        self.prior_ftr0 = nn.Parameter(torch.rand(self.dec_ch0, self.coarsest, self.coarsest), requires_grad=True)

        # ---- optional seed projection (RSSM feature -> coarsest decoder state)
        if self.feat_size is not None:
            self.seed_proj = nn.Sequential(
                nn.Linear(self.feat_size, self.dec_ch0 * self.coarsest * self.coarsest, bias=False),
                nn.LayerNorm(self.dec_ch0 * self.coarsest * self.coarsest, eps=1e-3),
                Act(),
            )
        else:
            self.seed_proj = None

        # spectral norm (optional)
        if cfg.get("spectral_norm", 0) > 0:
            fn = tools.AddNorm("spectral", (nn.Conv2d, nn.ConvTranspose2d), cfg["spectral_norm"], name="weight").get_fn()
            self.apply(fn)

    # ---- forward like original: returns (latents, feats, recon, q_all, p_all)
    def forward(self, x, t: float = 0.0, full: bool = False, seed=None):
        # normalize to NCHW
        had_time = (x.ndim == 5)   # [B,T,H,W,C]
        if had_time:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        elif x.ndim == 4 and x.shape[-1] in (1,2,3,4):
            x = x.permute(0, 3, 1, 2)
        elif x.ndim == 4 and x.shape[1] in (1,2,3,4):
            pass
        else:
            raise ValueError(f"CNVAE expects NHWC/NCHW 4D or 5D; got {tuple(x.shape)}")

        # enc stem + pre
        s = self.stem(x)
        for cell in self.pre_process:
            s = cell(s)

        # collect combiner taps as in original
        comb_enc, comb_s = [], []
        for cell in self.enc_tower:
            if isinstance(cell, CombinerEnc):
                comb_enc.append(cell)
                comb_s.append(s)
            else:
                s = cell(s)
        comb_enc.reverse()
        comb_s.reverse()

        # z0
        ftr_enc0 = self.enc0(s)
        param0   = self.enc_sampler[0](ftr_enc0)
        mu_q0, logs_q0 = param0.chunk(2, 1)
        dist_q0 = torchd.Independent(torchd.Normal(mu_q0, logs_q0.clamp(-6, 3).exp()), 1)
        z0 = dist_q0.rsample()
        q_all = [dist_q0]
        latents = [z0]

        # p(z0): standard normal (same shape as z0)
        mu0 = torch.zeros_like(z0)
        logs0 = torch.zeros_like(z0)
        dist_p0 = torchd.Independent(torchd.Normal(mu0, logs0.exp()), 1)
        p_all = [dist_p0]

        # decoder start: learned prior feature OR RSSM seed
        if seed is not None:
            if self.seed_proj is None:
                raise ValueError("CNVAE received a seed but cfg['feat_size'] was not set when constructing CNVAE.")
            if seed.ndim != 2:
                raise ValueError(f"Expected flat seed [N, feat_size], got {tuple(seed.shape)}")
            z = self.seed_proj(seed)
            s_dec = z.view(seed.shape[0], self.dec_ch0, self.coarsest, self.coarsest)
        else:
            N = z0.shape[0]
            s_dec = self.prior_ftr0.unsqueeze(0).expand(N, -1, -1, -1).contiguous()

        # run decoder tower; at each CombinerDec, compute prior/posterior for next z
        dec_id    = 0   # which CombinerDec / ladder-level we are on
        enc_id    = 0   # which enc combiner we use (for q at next levels)
        prior_id  = 0   # dec_sampler index (starts at 0 for level-1 prior)

        for cell in self.dec_tower:
            if isinstance(cell, CombinerDec):
                if dec_id == 0:
                    # combine z0 first
                    s_dec = cell(s_dec, self.expand[dec_id](latents[0]))
                else:
                    # form prior from decoder state
                    p_param = self.dec_sampler[prior_id](s_dec)
                    mu_p, logs_p = p_param.chunk(2, 1)
                    dist_p = torchd.Independent(torchd.Normal(mu_p, logs_p.clamp(-6, 3).exp()), 1)
                    p_all.append(dist_p)
                    prior_id += 1

                    # form posterior from encoder tap and current decoder state
                    enc_in  = comb_enc[enc_id](comb_s[enc_id], s_dec)
                    q_param = self.enc_sampler[dec_id](enc_in)
                    mu_q, logs_q = q_param.chunk(2, 1)
                    dist_q = torchd.Independent(torchd.Normal(mu_q, logs_q.clamp(-6, 3).exp()), 1)
                    z = dist_q.rsample()

                    q_all.append(dist_q)
                    latents.append(z)

                    # update decoder with new latent
                    s_dec = cell(s_dec, self.expand[dec_id](z))
                    enc_id += 1
                dec_id += 1
            else:
                s_dec = cell(s_dec)

        # post + out
        y = s_dec
        for cell in self.post_process:
            y = cell(y)
        recon = self.out(y)  # NCHW

        # back to NHWC/Dreamer conventions when full=True
        if full:
            if had_time:
                # recon: [B*T, C, H, W] -> [B,T,H,W,C]
                N, C, H, W = recon.shape
                recon = recon.view(B, T, C, H, W).permute(0, 1, 3, 4, 2)
            else:
                recon = recon.permute(0, 2, 3, 1)
            feats = None
            return latents, feats, recon, q_all, p_all
        else:
            return latents, None, None, q_all, p_all

    def xtract_ftr(self, x, t: float = 0.0, full: bool = False, seed=None):
        return self.forward(x, t, full=full, seed=seed)

    @torch.no_grad()
    def generate(self, z_list, seed=None):
        # start decoder state
        if seed is not None:
            if self.seed_proj is None:
                raise ValueError("CNVAE generate() got seed but no seed_proj (cfg['feat_size']).")
            s_dec = self.seed_proj(seed).view(seed.shape[0], self.dec_ch0, self.coarsest, self.coarsest)
        else:
            N = z_list[0].shape[0]
            s_dec = self.prior_ftr0.unsqueeze(0).expand(N, -1, -1, -1).contiguous()

        dec_id = 0
        for cell in self.dec_tower:
            if isinstance(cell, CombinerDec):
                s_dec = cell(s_dec, self.expand[dec_id](z_list[dec_id]))
                dec_id += 1
            else:
                s_dec = cell(s_dec)
        y = s_dec
        for cell in self.post_process:
            y = cell(y)
        recon = self.out(y)  # NCHW
        return recon.permute(0, 2, 3, 1), None

    @staticmethod
    def loss_kl(q_all, p_all):
        kl_all = []
        for q, p in zip(q_all, p_all):
            kl = torchd.kl.kl_divergence(q, p)
            # sum over channel and spatial (if present)
            while kl.ndim > 1:
                kl = kl.sum(-1)
            kl_all.append(kl)
        return kl_all, []


# ─────────────────────────────────────────────────────────────────────────────
#  Dreamer-torch RSSM exactly as original: supports discrete & gaussian
# ─────────────────────────────────────────────────────────────────────────────
class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
        # Poisson parameters
        use_poisson=False,
        poisson_temp=1.0,
    ):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._use_poisson = use_poisson
        self._poisson_temp = poisson_temp
        act = getattr(nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        # image→deter
        if self._use_poisson:
            inp_dim = stoch + num_actions  # Poisson latents are continuous
        else:
            inp_dim = stoch * (discrete if discrete else 1) + (0 if discrete else 0) + num_actions
            if not discrete:
                inp_dim = stoch + num_actions
        self._img_in_layers = nn.Sequential(
            nn.Linear(inp_dim, hidden, bias=False),
            *( [nn.LayerNorm(hidden, eps=1e-3)] if norm else [] ),
            act()
        )
        self._img_in_layers.apply(tools.weight_init)

        self._cell = GRUCell(hidden, deter, norm=norm)
        self._cell.apply(tools.weight_init)

        # deter→stats
        self._img_out_layers = nn.Sequential(
            nn.Linear(deter, hidden, bias=False),
            *( [nn.LayerNorm(hidden, eps=1e-3)] if norm else [] ),
            act()
        )
        self._img_out_layers.apply(tools.weight_init)

        # obs→posterior
        self._obs_out_layers = nn.Sequential(
            nn.Linear(deter + embed, hidden, bias=False),
            *( [nn.LayerNorm(hidden, eps=1e-3)] if norm else [] ),
            act()
        )
        self._obs_out_layers.apply(tools.weight_init)

        # stat layers
        if self._use_poisson:
            self._imgs_stat_layer = nn.Linear(hidden, stoch)
            self._obs_stat_layer  = nn.Linear(hidden, stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer .apply(tools.uniform_weight_init(1.0))
        elif discrete:
            self._imgs_stat_layer = nn.Linear(hidden, stoch * discrete)
            self._obs_stat_layer  = nn.Linear(hidden, stoch * discrete)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer .apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(hidden, 2 * stoch)
            self._obs_stat_layer  = nn.Linear(hidden, 2 * stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer .apply(tools.uniform_weight_init(1.0))

        if initial == "learned":
            self.W = nn.Parameter(torch.zeros((1, deter), device=device), requires_grad=True)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._use_poisson:
            state = dict(
                log_rate=torch.zeros(batch_size, self._stoch, device=self._device),
                stoch=torch.zeros(batch_size, self._stoch, device=self._device),
                deter=deter)
        elif self._discrete:
            state = dict(
                logit=torch.zeros(batch_size, self._stoch, self._discrete, device=self._device),
                stoch=torch.zeros(batch_size, self._stoch, self._discrete, device=self._device),
                deter=deter)
        else:
            state = dict(
                mean =torch.zeros(batch_size, self._stoch, device=self._device),
                std  =torch.zeros(batch_size, self._stoch, device=self._device),
                stoch=torch.zeros(batch_size, self._stoch, device=self._device),
                deter=deter)
        if self._initial == "zeros":
            return state
        if self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions), device=self._device)
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action = prev_action * (1.0 - is_first)
            init_state = self.initial(len(is_first))
            new_prev_state = {}
            for key, val in prev_state.items():
                is_first_r = torch.reshape(is_first, is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)))
                new_prev_state[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r
            prev_state = new_prev_state

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer("obs", x)
        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def img_step(self, prev_state, prev_action, sample=True):
        stoch = prev_state["stoch"]
        if self._discrete and not self._use_poisson:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        x = torch.cat([stoch, prev_action], -1)
        x = self._img_in_layers(x)

        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, [deter] = self._cell(x, [deter])
        x = self._img_out_layers(deter)

        stats = self._suff_stats_layer("ims", x)
        dist  = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        return {"deter": deter, **stats, "stoch": stoch}

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        return self.get_dist(stats).mode()

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete and not self._use_poisson:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def _suff_stats_layer(self, name, x):
        if name == "ims": layer = self._imgs_stat_layer
        elif name == "obs": layer = self._obs_stat_layer
        else: raise NotImplementedError
        out = layer(x)
        if self._use_poisson:
            return {"log_rate": out}
        elif self._discrete:
            logit = out.reshape(list(out.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        mean, std = out.chunk(2, -1)
        mean = {"none": lambda m: m, "tanh5": lambda m: 5 * torch.tanh(m / 5)}[self._mean_act](mean)
        std  = {"softplus": lambda s: F.softplus(s),
                "abs":      lambda s: s.abs() + 1,
                "sigmoid":  lambda s: torch.sigmoid(s),
                "sigmoid2": lambda s: 2 * torch.sigmoid(s / 2)}[self._std_act](std)
        std = std + self._min_std
        return {"mean": mean, "std": std}

    def get_dist(self, stats):
        if self._use_poisson:
            return Poisson(stats["log_rate"], self._poisson_temp)
        elif self._discrete:
            return torchd.independent.Independent(
                tools.OneHotDist(stats["logit"], unimix_ratio=self._unimix_ratio), 1
            )
        return tools.ContDist(
            torchd.independent.Independent(
                torchd.normal.Normal(stats["mean"], stats["std"]), 1
            )
        )

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale, step=None, free_bits_max=1.0, anneal_steps=50000):
        dist = lambda st: self.get_dist(st)
        sg   = lambda st: {k: v.detach() for k, v in st.items()}

        if self._use_poisson:
            rep = dist(post).kl(dist(sg(prior)))
            dyn = dist(sg(post)).kl(dist(prior))
        else:
            kld = torchd.kl.kl_divergence
            rep = kld(dist(post),     dist(sg(prior)))
            dyn = kld(dist(sg(post)), dist(prior))

        def reduce_free_bits(x):
            x = torch.clamp(x, min=free)
            if x.ndim > 2:
                x = x.sum(dim=-1)
            return x

        rep = reduce_free_bits(rep)
        dyn = reduce_free_bits(dyn)
        loss = dyn_scale * dyn + rep_scale * rep
        return loss, rep, dyn, rep, None


# ─────────────────────────────────────────────────────────────────────────────
#  MultiEncoder, MultiDecoder, ConvEncoder, ConvDecoder, MLP, GRUCell, ImgChLayerNorm
# ─────────────────────────────────────────────────────────────────────────────

class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(input_shape, cnn_depth, act, norm, kernel_size, minres)
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(input_size, None, mlp_layers, mlp_units, act, norm,
                            symlog_inputs=symlog_inputs, name="Encoder")
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(feat_size, shape, cnn_depth, act, norm, kernel_size, minres,
                                    outscale=outscale, cnn_sigmoid=cnn_sigmoid)
        if self.mlp_shapes:
            self._mlp = MLP(feat_size, self.mlp_shapes, mlp_layers, mlp_units, act, norm,
                            vector_dist, outscale=outscale, name="Decoder")
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            outputs = self._cnn(features)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._make_image_dist(output) for key, output in zip(self.cnn_shapes.keys(), outputs)})
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3))
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, depth=32, act="SiLU", norm=True, kernel_size=4, minres=4):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(Conv2dSamePad(in_dim, out_dim, kernel_size=kernel_size, stride=2, bias=False))
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(self, feat_size, shape=(3, 64, 64), depth=32, act=nn.ELU, norm=True,
                 kernel_size=4, minres=4, outscale=1.0, cnn_sigmoid=False):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 2,
                                             padding=(pad_h, pad_w),
                                             output_padding=(outpad_h, outpad_w),
                                             bias=bias))
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            out_dim //= 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        x = x.reshape([-1, self._minres, self._minres, self._embed_size // self._minres**2])
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        mean = x.reshape(features.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean = mean + 0.5
        return mean


class MLP(nn.Module):
    def __init__(self, inp_dim, shape, layers, units, act="SiLU", norm=True, dist="normal",
                 std=1.0, min_std=0.1, max_std=1.0, absmax=None, temp=0.1, unimix_ratio=0.01,
                 outscale=1.0, symlog_inputs=False, device="cuda", name="NoName"):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False))
            if norm:
                self.layers.add_module(f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03))
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for n, shape in self._shape.items():
                self.mean_layer[n] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for n, shape in self._shape.items():
                    self.std_layer[n] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for n, shape in self._shape.items():
                mean = self.mean_layer[n](out)
                std  = self.std_layer[n](out) if self._std == "learned" else self._std
                dists.update({n: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            std  = self.std_layer(out) if self._std == "learned" else self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, tools.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = tools.ContDist(torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax)
        elif dist == "huber":
            dist = tools.ContDist(torchd.independent.Independent(
                tools.UnnormalizedHuber(mean, std, 1.0), len(shape), absmax=self._absmax
            ))
        elif dist == "binary":
            dist = tools.Bernoulli(torchd.independent.Independent(
                torchd.bernoulli.Bernoulli(logits=mean), len(shape)
            ))
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module("GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False))
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
