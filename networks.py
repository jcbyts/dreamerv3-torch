import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools

#-----------------------------------------------------------------
# Original RSSM
# ----------------------------------------------------------------
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
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior


    def imagine_with_action(self, action, state):
        # action: (B,T,A), state: dict of current posterior (B, ...)
        swap = lambda x: x.permute([1,0] + list(range(2, x.ndim)))
        action_T = swap(action)

        def step(prev, a):
            prior, spatial = self.img_step(prev, a, sample=True)
            return prior, (prior, spatial)

        (priors_T, spatials_T), _ = tools.static_scan(step, [action_T], state)
        prior_seq   = {k: (v if not isinstance(v, list) else [swap(x) for x in v]) for k, v in priors_T.items()}
        spatial_seq = swap(spatials_T[1])  # (B,T,C,H,W)
        return prior_seq, spatial_seq

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


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
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
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
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
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
        obs -= 0.5
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
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
        h, w = minres, minres
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
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
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
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
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
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
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
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
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
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


#-----------------------------------------------------------------
# New hRSSM
# ----------------------------------------------------------------
class BlockDiagGRUCell(nn.Module):
    """
    Block-diagonal GRU cell as provided. Each block will correspond to a
    level in the hierarchy, allowing for parallel state updates.
    """
    def __init__(self, inp_size, size, blocks, norm=True, act=torch.tanh, update_bias=-1):
        super().__init__()
        assert size % blocks == 0
        self.size, self.blocks, self.b = size, blocks, size // blocks
        self.act, self.update_bias = act, update_bias

        self.Wx = nn.Linear(inp_size, 3*size, bias=False)
        self.Wrec = nn.Parameter(torch.empty(blocks, self.b, 3*self.b))
        nn.init.xavier_uniform_(self.Wrec)

        self.ln = nn.LayerNorm(3*size, eps=1e-3) if norm else None

    def forward(self, x, state):
        h = state[0]
        N = h.shape[0]
        hB = h.view(N, self.blocks, self.b)

        parts = self.Wx(x)
        parts_rec = torch.einsum('nkb,kbc->nkc', hB, self.Wrec).reshape(N, 3*self.size)
        parts = parts + parts_rec
        if self.ln is not None:
            parts = self.ln(parts)

        r, c, z = torch.split(parts, self.size, dim=-1)
        r = torch.sigmoid(r)
        c = self.act(r * c)
        z = torch.sigmoid(z + self.update_bias)
        h_new = z * c + (1 - z) * h
        return h_new, [h_new]
    
class _ProductDist:
    """Treats a list of independent distributions as one big factored dist."""
    def __init__(self, dists):
        self._dists = dists                      # list length = h_levels

    def log_prob(self, x_list):
        # x_list is a list with same length as self._dists
        return sum(d.log_prob(x) for d, x in zip(self._dists, x_list))

    def entropy(self):
        return sum(d.entropy() for d in self._dists)

    def sample(self):
        return [d.rsample() if hasattr(d, 'rsample') else d.sample()
                for d in self._dists]

    def mode(self):
        return [d.mean if hasattr(d, "mean") else d.mode()
                for d in self._dists]


class hConvEncoder(nn.Module):
    """
    Returns a list of feature maps (finest -> coarsest) with exactly `levels` items.
    Spatial sizes are: [minres*2^(levels-1), ..., minres*2, minres].
    """
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
        levels=3,
        grow_channels=False,
        **kwargs,  # ignore extra keys to stay drop-in compatible
    ):
        super().__init__()
        act = getattr(torch.nn, act)
        h, w, in_ch = input_shape
        assert h == w, "This encoder assumes square inputs."
        assert (h % minres) == 0, "Input size must be divisible by minres by powers of 2."

        # Compute how many total stride-2 downs we need to reach minres
        hh = h
        total_downs = 0
        while hh > minres:
            hh //= 2
            total_downs += 1
        assert hh == minres, "h/minres must be a power of 2."
        assert levels <= total_downs, (
            f"levels={levels} is too large for input {h} and minres {minres}; "
            f"max levels is {total_downs}."
        )

        # Number of early downsamples BEFORE we start emitting level features
        pre_downs = total_downs - levels

        # Build pre-downsampling stack (stem). These do NOT produce outputs.
        pre = nn.ModuleList()
        self.out_channels = []

        curr_in  = in_ch
        curr_out = depth
        for i in range(pre_downs):
            
            block = nn.Sequential(
                Conv2dSamePad(curr_in, curr_out, kernel_size=kernel_size, stride=2, bias=False),
                ImgChLayerNorm(curr_out) if norm else nn.Identity(),
                act(),
            )
            pre.append(block)
            curr_in = curr_out
            if grow_channels:
                curr_out *= 2
        
        self.stem = nn.Sequential(*pre)

        # Build the `levels` downsampling blocks. Each produces one output.
        self.levels = nn.ModuleList()
        for l in range(levels):
            self.out_channels.append(curr_out)
            block = nn.Sequential(
                Conv2dSamePad(curr_in, curr_out, kernel_size=kernel_size, stride=2, bias=False),
                ImgChLayerNorm(curr_out) if norm else nn.Identity(),
                act(),
            )
            self.levels.append(block)
            curr_in = curr_out
            if grow_channels:
                curr_out *= 2

        self.apply(tools.weight_init)

    def forward(self, obs):
        """
        obs: (B, T, H, W, C) in [0,1]
        returns: list of (B, T, C, H, W), length == `levels`, finest -> coarsest
        """
        x = obs["image"] - 0.5                                # (B,T,H,W,C)
        sz = x.shape
        H = sz[-3]
        W = sz[-2]
        C = sz[-1]

        # Handle different input shapes
        if x.ndim == 3:  # (H, W, C) - raw environment observation
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and time dims: (B=1, T=1, H, W, C)
            B = T = 1
        elif x.ndim == 4:  # (B, H, W, C) - evaluation case
            x = x.unsqueeze(1)  # Add time dim: (B, T=1, H, W, C)
            B = sz[0]
            T = 1
        elif x.ndim == 5:  # (B, T, H, W, C) - training case
            B = sz[0]
            T = sz[1]
            pass  # Already correct shape
        else:
            raise ValueError(f"Expected 3D, 4D or 5D image tensor, got {x.ndim}D")
        
        BT = B * T
        x = x.reshape(BT, H, W, C)                       # (BT,H,W,C)
        x = x.permute(0, 3, 1, 2)                             # (BT,C,H,W)

        # Pre-levels (no outputs collected)
        x = self.stem(x)

        feats = []
        # Level blocks (collect outputs)
        for block in self.levels:
            x = block(x)                                      # (BT,C,H,W)
            f = x.reshape([B, T] + list(x.shape[1:]))  # (B,T,C,H,W)
            feats.append(f)

        return feats

class ExpandZ(nn.Module):
    """
    Expand a flat latent z to a spatial tensor via a single ConvTranspose2d.
    Input:  z as (B, Cin) or (B, Cin, 1, 1)
    Output: (B, Cout, H, H)
    """
    def __init__(self, in_ch, out_ch, spatial, norm=True, act="SiLU"):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=spatial,  # 1x1 -> HxH in one shot
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm = ImgChLayerNorm(out_ch) if norm else nn.Identity()
        self.act = getattr(nn, act)()

    def forward(self, z):
        # z: (B, C) or (B, C, 1, 1)
        if z.dim() == 2:
            z = z[:, :, None, None]
        x = self.deconv(z)
        x = self.norm(x)
        x = self.act(x)
        return x


class Up2x(nn.Module):
    """
    Learned 2x upsampling block that preserves channels.
    Upsample -> 1x1 conv -> norm -> act
    """
    def __init__(self, channels, mode="bilinear", norm=True, act="SiLU"):
        super().__init__()
        # align_corners only applies to some modes
        align = (mode in ("bilinear", "bicubic"))
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False if align else None)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = ImgChLayerNorm(channels) if norm else nn.Identity()
        self.act = getattr(nn, act)()

    def forward(self, x):
        x = self.up(x)
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class CompressShared(nn.Module):
    """
    Shared compression head (learned pooling) used by both prior and posterior.
    - core input: required (e.g., broadcast deter_l concatenated with parent feature)
    - enc input: optional encoder feature at the same spatial resolution
    Projects core and enc to the same hidden width, sums (when enc is present),
    then norm+act, then a learned global pooling conv with kernel_size=H, then FC to stats.

    Forward returns a flat vector of size out_dim so the caller can split into
    mean/std or logits.
    """
    def __init__(self, core_in_ch, enc_in_ch, hidden_ch, spatial, out_dim,
                 norm=True, act="SiLU"):
        super().__init__()
        self.has_enc = enc_in_ch > 0

        self.core_proj = nn.Conv2d(core_in_ch, hidden_ch, kernel_size=1, bias=False)
        self.enc_proj  = nn.Conv2d(enc_in_ch, hidden_ch, kernel_size=1, bias=False) if self.has_enc else None

        self.norm = ImgChLayerNorm(hidden_ch) if norm else nn.Identity()
        self.act  = getattr(nn, act)()

        # Learned global pooling: kernel_size = spatial H (assumed square HxH)
        self.pool = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=spatial, stride=1, padding=0, bias=False)

        self.flat = nn.Flatten()
        self.fc   = nn.Linear(hidden_ch, out_dim)

    def forward(self, core, enc=None):
        """
        core: (B, C_core, H, W)
        enc : (B, C_enc,  H, W) or None
        """
        h = self.core_proj(core)
        if self.has_enc and enc is not None:
            # Shapes must match spatially already (H, W)
            e = self.enc_proj(enc)
            h = h + e
        h = self.norm(h)
        h = self.act(h)
        h = self.pool(h)      # -> (B, hidden_ch, 1, 1)
        h = self.flat(h)      # -> (B, hidden_ch)
        h = self.fc(h)        # -> (B, out_dim)
        return h
    
class hRSSM(RSSM):
    def __init__(
        self,
        h_levels=3,
        h_stoch_dims=[32, 32, 32],
        h_deter_dims=[128, 256, 256],
        h_hidden_dim=200,
        h_encoder_dims=[128, 256, 128],
        act="SiLU",
        norm=True,
        up_mode="nearest",
        **kwargs
    ):
        nn.Module.__init__(self)
        # ----- core config copied from RSSM ---------------------------------
        self._discrete     = kwargs.get("discrete", False)  # K categories or False
        self._unimix_ratio = kwargs.get("unimix_ratio", 0.01)
        self._initial      = kwargs.get("initial", "learned")
        self._num_actions  = kwargs.get("num_actions")
        self._device       = kwargs.get("device")
        self._mean_act     = kwargs.get("mean_act", "none")
        self._std_act      = kwargs.get("std_act", "sigmoid2")
        self._min_std      = kwargs.get("min_std", 0.1)
        self._expose_levels = kwargs.get("expose_levels", "all")

        # ----- hierarchy sizes ---------------------------------------------
        self._h_levels       = h_levels
        self._h_stoch_dims   = h_stoch_dims         # per level, finest -> coarsest
        self._h_deter_dims   = h_deter_dims         # per level, finest -> coarsest
        self._h_hidden_dim   = h_hidden_dim
        self._h_encoder_dims = h_encoder_dims       # encoder C per level, finest -> coarsest

        assert len(self._h_stoch_dims)   == self._h_levels
        assert len(self._h_deter_dims)   == self._h_levels
        assert len(self._h_encoder_dims) == self._h_levels

        self._deter = sum(self._h_deter_dims)
        act_fn = getattr(torch.nn, act)

        # ----- spatial sizes per level (finest -> coarsest) ----------------
        # We infer a consistent pyramid from minres and h_levels.
        # Example: minres=4, h_levels=3  -> [16, 8, 4]
        minres = kwargs.get("minres", 4)
        L = self._h_levels
        self._spatial_sizes = [minres * (2 ** (L - 1 - i)) for i in range(L)]  # finest -> coarsest

        # ----- GRU input (flat stoch + actions) -----------------------------
        K = self._discrete if self._discrete else 1
        if self._discrete:
            inp_dim = sum(s * K for s in self._h_stoch_dims)
        else:
            inp_dim = sum(self._h_stoch_dims)
        inp_dim += self._num_actions

        self._img_in_layers = nn.Sequential(
            nn.Linear(inp_dim, self._h_hidden_dim, bias=False),
            nn.LayerNorm(self._h_hidden_dim, eps=1e-3) if norm else nn.Identity(),
            act_fn(),
        )

        # Block-diagonal GRU (deterministic state)
        self._cell = BlockDiagGRUCell(
            self._h_hidden_dim, self._deter,
            blocks=self._h_levels, norm=norm, act=torch.tanh
        )

        # ===================================================================
        # CNVAE-style spatial path modules (shared by prior and posterior)
        # ===================================================================

        # 1) channels for flat z and for its spatial expansion per level
        #    Use Cz[l] = channels after expanding z_l to (H[l], H[l]).
        self._z_flat_ch = [s * K for s in self._h_stoch_dims]           # (B, s*K) if discrete else (B, s)
        Cz        = self._z_flat_ch                                     # simple, effective choice

        # 2) core input channels seen by the compressor at each level:
        #    core = concat( broadcast(deter_l), parent_feat )
        core_in_ch = []
        for l in range(L):
            parent_ch = Cz[l + 1] if (l < L - 1) else 0
            core_in_ch.append(self._h_deter_dims[l] + parent_ch)

        # 3) output dim (before splitting into mean/std or logits)
        out_dim = [ (s * K) if (K > 1) else (2 * s) for s in self._h_stoch_dims ]

        # 4) ExpandZ: turn flat z_l into spatial Cz[l] x H[l] x H[l]
        self.expand_z = nn.ModuleList([
            ExpandZ(
                in_ch=self._z_flat_ch[l],
                out_ch=Cz[l],
                spatial=self._spatial_sizes[l],
                norm=norm,
                act=act,
            )
            for l in range(L)
        ])

        # 5) Up2x for transitions coarse->fine: l+1 -> l (there are L-1 of these)
        #    Channels of parent feature are Cz[l+1].
        self.up2x = nn.ModuleList([
            Up2x(
                channels=Cz[l + 1],
                mode=up_mode,
                norm=norm,
                act=act,
            )
            for l in range(L - 1)
        ])

        # 6) Shared learned compressor per level (prior: enc=None, posterior: enc=feat)
        #    enc_in_ch is the encoder channels at that level.
        self.compress = nn.ModuleList([
            CompressShared(
                core_in_ch=core_in_ch[l],
                enc_in_ch=self._h_encoder_dims[l],
                hidden_ch=self._h_hidden_dim,
                spatial=self._spatial_sizes[l],
                out_dim=out_dim[l],
                norm=norm,
                act=act,
            )
            for l in range(L)])

        # ----- learned initial deter ---------------------------------------
        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

        # ----- init weights -------------------------------------------------
        self.apply(tools.weight_init)
        print("hRSSM (shared spatial compressor + expand_z + up2x) initialised.")

    
    def feat_size(self):
        """
        Calculate the number of channels in the feature output
        """
        # Each level contributes (deter_dim + stoch_dim) channels
        total_channels = 0
        if self._expose_levels == 'all':
            for i in range(self._h_levels):
                level_channels = self._h_deter_dims[i] + self._z_flat_ch[i]
                total_channels += level_channels
        elif self._expose_levels == 'top':
            for i in [self._h_levels-1]:
                level_channels = self._h_deter_dims[i] + self._z_flat_ch[i]
                total_channels += level_channels
        else:
            i = int(self._expose_levels)
            level_channels = self._h_deter_dims[i] + self._z_flat_ch[i]
            total_channels += level_channels

        return total_channels
    
    def _flat_z(self, z):
        return z.flatten(start_dim=1) if self._discrete and z.ndim == 3 else z
    
    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._initial == "learned":
            deter = torch.tanh(self.W).repeat(batch_size, 1)
        if self._discrete:
            state = {
                'logit': [torch.zeros([batch_size, d, self._discrete], device=self._device) for d in self._h_stoch_dims],
                'stoch': [torch.zeros([batch_size, d, self._discrete], device=self._device) for d in self._h_stoch_dims],
                'deter': deter,
            }
        else:
            state = {
                'mean': [torch.zeros([batch_size, d], device=self._device) for d in self._h_stoch_dims],
                'std': [torch.ones([batch_size, d], device=self._device) for d in self._h_stoch_dims],
                'stoch': [torch.zeros([batch_size, d], device=self._device) for d in self._h_stoch_dims],
                'deter': deter,
            }
        return state
    
    def img_step(self, prev_state, prev_action, sample=True):
        """
        Prior ladder (coarse -> fine) using shared compressor without encoder.
        Returns:
        prior  : dict with per-level stats and samples (same ordering as levels: 0..L-1 finest->coarsest)
        spatial: composed spatial feature map built from deter and prior stoch
        """
        B = prev_action.shape[0]
        K = self._discrete if self._discrete else 1
        L = self._h_levels

        # 1) deterministic update (GRU)
        stoch_list_flat = [self._flat_z(s) for s in prev_state["stoch"]]  # list of (B, s*K) or (B,s)
        prev_stoch_flat = torch.cat(stoch_list_flat, dim=1)               # (B, sum s*K) or (B,sum s)
        x = torch.cat([prev_stoch_flat, prev_action], dim=-1)
        x = self._img_in_layers(x)
        deter, _ = self._cell(x, [prev_state["deter"]])                   # (B, sum deter)
        deter_split = torch.split(deter, self._h_deter_dims, dim=-1)      # list l=0..L-1 finest->coarsest

        # 2) containers
        prior_stoch = [None] * L
        if self._discrete:
            prior_stats = {"logit": []}
        else:
            prior_stats = {"mean": [], "std": []}

        # 3) level L-1 (coarsest)
        l = L - 1
        H = self._spatial_sizes[l]
        deter_img = deter_split[l].view(B, -1, 1, 1).expand(B, -1, H, H)  # (B, D_l, H, H)
        core = deter_img                                                   # no parent feature at coarsest
        stats_vec = self.compress[l](core, enc=None)                       # (B, out_dim_l)

        # parse stats -> dist -> sample z_l (flat)
        if self._discrete:
            logit = stats_vec.view(B, self._h_stoch_dims[l], K)
            stats = {"logit": logit}
        else:
            mean, std = torch.split(stats_vec, self._h_stoch_dims[l], dim=-1)
            mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
            std  = {
                "softplus": lambda: torch.nn.functional.softplus(std),
                "abs":      lambda: torch.abs(std + 1),
                "sigmoid":  lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]() + self._min_std
            stats = {"mean": mean, "std": std}
        z = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
        prior_stoch[l] = z
        for k, v in stats.items():
            prior_stats.setdefault(k, [])
            prior_stats[k].insert(0 if l == 0 else 0, v)  # maintain list shape; exact position not used later by name

        # spatial parent for next (finer) level: expand z_l to (H,H), then up2x when moving down
        parent_spatial = self.expand_z[l](self._flat_z(z))                 # (B, Cz_l, H, H)

        # 4) levels L-2 ... 0 (coarse -> fine)
        for l in range(L - 2, -1, -1):
            # upsample parent feature to current level resolution
            parent_spatial = self.up2x[l](parent_spatial)                  # (B, Cz_{l+1}, H_l, H_l)

            H = self._spatial_sizes[l]
            deter_img = deter_split[l].view(B, -1, 1, 1).expand(B, -1, H, H)   # (B, D_l, H, H)
            core = torch.cat([deter_img, parent_spatial], dim=1)               # (B, D_l + Cz_{l+1}, H, H)

            stats_vec = self.compress[l](core, enc=None)                       # (B, out_dim_l)
            if self._discrete:
                logit = stats_vec.view(B, self._h_stoch_dims[l], K)
                stats = {"logit": logit}
            else:
                mean, std = torch.split(stats_vec, self._h_stoch_dims[l], dim=-1)
                mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
                std  = {
                    "softplus": lambda: torch.nn.functional.softplus(std),
                    "abs":      lambda: torch.abs(std + 1),
                    "sigmoid":  lambda: torch.sigmoid(std),
                    "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
                }[self._std_act]() + self._min_std
                stats = {"mean": mean, "std": std}

            z = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
            prior_stoch[l] = z
            for k, v in stats.items():
                prior_stats[k].insert(0, v)

            # refresh parent feature for the next finer level
            parent_spatial = self.expand_z[l](self._flat_z(z))                 # (B, Cz_l, H, H)

        prior = {"stoch": prior_stoch, "deter": deter, **prior_stats}

        return prior, parent_spatial

    def obs_step(self, prev_state, prev_action, embed_list, is_first, sample=True):
        """
        Posterior ladder (coarse -> fine) using shared compressor with encoder features.
        Returns:
        post, prior, spatial_post
        """
        # Handle episode starts
        B = embed_list[0].shape[0] # should work regardless of whether prev_action or perv_state are none
        if prev_state is None or torch.all(is_first):
            prev_state = self.initial(B)
            prev_action = torch.zeros(B, self._num_actions, device=self._device)
        elif torch.any(is_first):
            
            prev_action_mask = (1 - is_first).float().unsqueeze(-1)
            prev_action = prev_action * prev_action_mask
            
            init_state = self.initial(B)
            for k, v in prev_state.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        # Dynamically reshape the mask for each tensor
                        mask = (1 - is_first).float()
                        mask = mask.reshape(mask.shape + (1,) * (v[i].ndim - mask.ndim))
                        prev_state[k][i] = v[i] * mask + init_state[k][i] * (1 - mask)
                else:
                    # dynamic mask to the 'deter' tensor
                    mask = (1 - is_first).float()
                    mask = mask.reshape(mask.shape + (1,) * (v.ndim - mask.ndim))
                    prev_state[k] = v * mask + init_state[k] * (1 - mask)

        # Utility: ensure encoder feats are (B, C, H, W)
        def as_spatial(feat):
            return feat[:, 0] if feat.ndim == 5 else feat

        embeds = [as_spatial(e) for e in embed_list]  # finest -> coarsest
        L = self._h_levels
        K = self._discrete if self._discrete else 1

        # Prior for KL (no gradients blocked; we manage stop-grads in kl_loss)
        prior, _ = self.img_step(prev_state, prev_action, sample=sample)
        deter_split = torch.split(prior["deter"], self._h_deter_dims, dim=-1)

        # Containers for posterior
        post_stoch = [None] * L
        if self._discrete:
            post_stats = {"logit": []}
        else:
            post_stats = {"mean": [], "std": []}

        # Level L-1 (coarsest)
        l = L - 1
        enc_feat = embeds[l]                                         # (B, E_l, H, H)
        H = enc_feat.shape[-1]
        deter_img = deter_split[l].view(B, -1, 1, 1).expand(B, -1, H, H)
        core = deter_img                                             # no parent feature at coarsest
        stats_vec = self.compress[l](core, enc=enc_feat)             # (B, out_dim_l)

        # parse -> dist -> sample
        if self._discrete:
            logit = stats_vec.view(B, self._h_stoch_dims[l], K)
            stats = {"logit": logit}
        else:
            mean, std = torch.split(stats_vec, self._h_stoch_dims[l], dim=-1)
            mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
            std  = {
                "softplus": lambda: torch.nn.functional.softplus(std),
                "abs":      lambda: torch.abs(std + 1),
                "sigmoid":  lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]() + self._min_std
            stats = {"mean": mean, "std": std}
        z = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
        post_stoch[l] = z
        for k, v in stats.items():
            post_stats.setdefault(k, [])
            post_stats[k].insert(0 if l == 0 else 0, v)

        parent_spatial = self.expand_z[l](self._flat_z(z))           # (B, Cz_l, H, H)

        # Levels L-2 ... 0 (coarse -> fine)
        for l in range(L - 2, -1, -1):
            # upsample parent posterior feature to current level
            parent_spatial = self.up2x[l](parent_spatial)            # (B, Cz_{l+1}, H_l, H_l)

            enc_feat = embeds[l]                                     # (B, E_l, H_l, H_l)
            H = enc_feat.shape[-1]
            deter_img = deter_split[l].view(B, -1, 1, 1).expand(B, -1, H, H)

            core = torch.cat([deter_img, parent_spatial], dim=1)     # (B, D_l + Cz_{l+1}, H, H)
            stats_vec = self.compress[l](core, enc=enc_feat)         # (B, out_dim_l)

            if self._discrete:
                logit = stats_vec.view(B, self._h_stoch_dims[l], K)
                stats = {"logit": logit}
            else:
                mean, std = torch.split(stats_vec, self._h_stoch_dims[l], dim=-1)
                mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
                std  = {
                    "softplus": lambda: torch.nn.functional.softplus(std),
                    "abs":      lambda: torch.abs(std + 1),
                    "sigmoid":  lambda: torch.sigmoid(std),
                    "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
                }[self._std_act]() + self._min_std
                stats = {"mean": mean, "std": std}

            z = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
            post_stoch[l] = z
            for k, v in stats.items():
                post_stats[k].insert(0, v)

            parent_spatial = self.expand_z[l](self._flat_z(z))       # (B, Cz_l, H, H)

        # Package outputs
        if self._discrete:
            post_stats["stoch"] = post_stoch
        post = {"stoch": post_stoch, "deter": prior["deter"], **post_stats}

        return post, prior, parent_spatial


    def observe(self, embed_list, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, x.ndim)))
        embed_list_T = [swap(e) for e in embed_list]
        action_T, is_first_T = swap(action), swap(is_first)

        def step(prev, a, embeds, first):
            post, prior, spatial = self.obs_step(prev[0], a, embeds, first, sample=True)
            return post, prior, spatial

        results = tools.static_scan(
            step, (action_T, list(zip(*embed_list_T)), is_first_T), (state, state, None)
        )
        post_seq_T, prior_seq_T, spatial_seq_T = results

        unswap = lambda t: t.permute([1, 0] + list(range(2, t.ndim)))
        post_seq  = {k: (unswap(v) if not isinstance(v, list) else [unswap(x) for x in v]) for k, v in post_seq_T.items()}
        prior_seq = {k: (unswap(v) if not isinstance(v, list) else [unswap(x) for x in v]) for k, v in prior_seq_T.items()}
        spatial_seq = unswap(spatial_seq_T)
        return post_seq, prior_seq, spatial_seq
    
    def imagine_with_action(self, action, state, sample=True):
        swap = lambda x: x.permute([1, 0] + list(range(2, x.ndim)))
        action_T = swap(action)

        # def step(prev_state, act_t):
        #     prior_t, spatial_t = self.img_step(prev_state, act_t, sample=sample)
        #     return prior_t, spatial_t

        # results = tools.static_scan(step, [action_T], state)
        def step(prev_state, act_t):
            # 'prev_state' is now always a tuple, so we can reliably get the state dict from index 0.
            state_dict = prev_state[0]
            prior_t, spatial_t = self.img_step(state_dict, act_t, sample=sample)
            return prior_t, spatial_t
        
        results = tools.static_scan(step, [action_T], (state,))
        priors_T, spatials_T = results

        unswap = lambda t: t.permute([1, 0] + list(range(2, t.ndim)))
        prior_seq = {
            k: (unswap(v) if not isinstance(v, list) else [unswap(x) for x in v])
            for k, v in priors_T.items()
        }
        spatial_seq = unswap(spatials_T)
        return prior_seq, spatial_seq
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        sg = lambda x: {k: v.detach() if isinstance(v, torch.Tensor) else [vi.detach() for vi in v] for k, v in x.items()}

        dyn_loss_list, rep_loss_list, kl_value_list = [], [], []

        for i in range(self._h_levels):
            post_dist = self.get_dist_h({k: v[i] for k, v in post.items()})
            prior_dist = self.get_dist_h({k: v[i] for k, v in prior.items()})
            prior_dist_sg = self.get_dist_h({k: v[i] for k, v in sg(prior).items()})
            post_dist_sg = self.get_dist_h({k: v[i] for k, v in sg(post).items()})
            
            dyn_loss_list.append(kld(post_dist_sg, prior_dist))
            rep_loss_list.append(kld(post_dist, prior_dist_sg))
            kl_value_list.append(kld(post_dist, prior_dist))
        
        dyn_loss = torch.stack(dyn_loss_list, dim=0).sum(0)
        rep_loss = torch.stack(rep_loss_list, dim=0).sum(0)
        kl_value = torch.stack(kl_value_list, dim=0).sum(0)

        dyn_loss = torch.clip(dyn_loss, min=free)
        rep_loss = torch.clip(rep_loss, min=free)
        
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, kl_value, dyn_loss, rep_loss

    def get_feat(self, state):
        stoch_list = state['stoch']
        stoch_list_flat = [self._flat_z(s) for s in stoch_list]

        if self._expose_levels == 'all':
            stoch_flat = torch.cat(stoch_list_flat, -1)
            return torch.cat([stoch_flat, state['deter']], -1)
        elif self._expose_levels == 'top':
            return torch.cat([stoch_list_flat[-1], torch.split(state['deter'], self._h_deter_dims, dim=-1)[-1]], -1)
        else:
            level_idx = int(self._expose_levels)
            return torch.cat([stoch_list_flat[level_idx], torch.split(state['deter'], self._h_deter_dims, dim=-1)[level_idx]], -1)
    
    def get_dist_h(self, state_level):
        if self._discrete:
            logit = state_level["logit"]
            # The Independent wrapper is the key fix. It sums over the last dimension.
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
            return dist
        else:
            mean, std = state_level["mean"], state_level["std"]
            return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        
    # def get_dist_h(self, state_level):
    #     if self._discrete:
    #         logit = state_level["logit"]
    #         return tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio)
    #     else:
    #         mean, std = state_level["mean"], state_level["std"]
    #         return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
    
    def get_dist(self, full_state):
        dists = [self.get_dist_h({k: (v[i] if isinstance(v, list) else v)
                                for k, v in full_state.items()})
                for i in range(self._h_levels)]
        return _ProductDist(dists)

    # def _suff_stats_layer_h(self, layer, x, stoch_dim):
    #     stats = layer(x)
    #     if stats.ndim == 4:
    #         stats = stats.flatten(start_dim=1)
    #     if self._discrete:
    #         logit = stats.view(stats.shape[0], stoch_dim, self._discrete)
    #         return {"logit": logit}
    #     else:
    #         mean, std = torch.split(stats, stoch_dim, dim=-1)
    #         mean = {
    #             "none": lambda: mean,
    #             "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
    #         }[self._mean_act]()
    #         std = {
    #             "softplus": lambda: torch.nn.functional.softplus(std),
    #             "abs": lambda: torch.abs(std + 1),
    #             "sigmoid": lambda: torch.sigmoid(std),
    #             "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
    #         }[self._std_act]() + self._min_std
    #         return {"mean": mean, "std": std}
