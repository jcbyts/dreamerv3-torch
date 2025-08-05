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
    Same API as ConvEncoder, but returns a list of feature maps:
    [stem(H/2), block1(H/4), ..., last(>=minres)] (finest -> coarsest).
    """
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        super().__init__()
        act = getattr(torch.nn, act)
        h, w, in_ch = input_shape

        # --- Stem ---
        stem = [
            Conv2dSamePad(in_ch, depth, kernel_size=kernel_size, stride=2, bias=False),
            ImgChLayerNorm(depth) if norm else nn.Identity(),
            act(),
        ]
        self.stem = nn.Sequential(*stem)

        # --- Blocks until >= minres ---
        self.blocks = nn.ModuleList()
        in_dim = depth
        out_dim = depth * 2

        # estimate sizes after the stem first
        hh, ww = h // 2, w // 2
        while min(hh, ww) > minres:
            block = [
                Conv2dSamePad(in_dim, out_dim, kernel_size=kernel_size, stride=2, bias=False),
                ImgChLayerNorm(out_dim) if norm else nn.Identity(),
                act(),
            ]
            self.blocks.append(nn.Sequential(*block))
            in_dim = out_dim
            out_dim *= 2
            hh //= 2
            ww //= 2

        self.apply(tools.weight_init)

    def forward(self, obs):
        """
        obs: (batch, time, H, W, C) float in [0,1] (matches ConvEncoder expectation).
        Returns: list of tensors shaped (batch, time, C, H, W), finest -> coarsest.
        """
        # Match ConvEncoder preprocessing but avoid in-place mutation.
        x = obs - 0.5

        # (B,T,H,W,C) -> (B*T,C,H,W)
        x = x.reshape((-1,) + tuple(x.shape[-3:]))  # (BT,H,W,C)
        x = x.permute(0, 3, 1, 2)                   # (BT,C,H,W)

        feats = []

        # Stem (H/2)
        x = self.stem(x)
        stem_feat = x.reshape(list(obs.shape[:-3]) + list(x.shape[-3:]))  # (B,T,C,H,W)
        feats.append(stem_feat)

        # Pyramid blocks (H/4, H/8, ...)
        for block in self.blocks:
            x = block(x)
            f = x.reshape(list(obs.shape[:-3]) + list(x.shape[-3:]))
            feats.append(f)

        return feats  # finest -> coarsest

class hRSSM(RSSM):
    """
    Hierarchical Recurrent State-Space Model.
    Manages temporal recurrence and the hierarchical generative process.
    """
    def __init__(
        self,
        h_levels=3,
        h_stoch_dims=[32, 32, 32],
        h_deter_dims=[128, 256, 256],
        h_hidden_dim=200,
        h_encoder_dims=[128, 256, 128], # Corresponds to hConvEncoder output channels
        act="SiLU",
        norm=True,
        up_mode="nearest",
        **kwargs
    ):
        # We call nn.Module.__init__ directly to bypass RSSM's original __init__
        nn.Module.__init__(self)
        
        # Copy necessary attributes from original RSSM that are still needed
        self._discrete = kwargs.get('discrete', False)
        self._unimix_ratio = kwargs.get('unimix_ratio', 0.01)
        self._initial = kwargs.get('initial', 'learned')
        self._num_actions = kwargs.get('num_actions')
        self._device = kwargs.get('device')
        self._mean_act = kwargs.get('mean_act', 'none')
        self._std_act = kwargs.get('std_act', 'sigmoid2')
        self._min_std = kwargs.get('min_std', 0.1)

        # Hierarchical parameters
        self._h_levels = h_levels
        self._h_stoch_dims = h_stoch_dims
        self._h_deter_dims = h_deter_dims
        self._h_hidden_dim = h_hidden_dim
        self._h_encoder_dims = h_encoder_dims
        
        assert len(self._h_stoch_dims) == self._h_levels
        assert len(self._h_deter_dims) == self._h_levels
        assert len(self._h_encoder_dims) == self._h_levels

        self._deter = sum(h_deter_dims)
        act_fn = getattr(torch.nn, act)
        
        # Input layer for GRU
        inp_dim = sum(self._h_stoch_dims) if not self._discrete else sum(s * d for s, d in zip(self._h_stoch_dims, [self._discrete]*self._h_levels))
        inp_dim += self._num_actions
        
        self._img_in_layers = nn.Sequential(
            nn.Linear(inp_dim, self._h_hidden_dim, bias=False),
            nn.LayerNorm(self._h_hidden_dim, eps=1e-03) if norm else nn.Identity(),
            act_fn()
        )
        
        self._cell = BlockDiagGRUCell(
            self._h_hidden_dim, self._deter, blocks=self._h_levels, norm=norm, act=torch.tanh
        )

        # Posterior and Prior networks for each level
        self.obs_stat_layers = nn.ModuleList()
        self.imgs_stat_layers = nn.ModuleList()
        self.expand_modules = nn.ModuleList()
        
        # Coarsest level (top of hierarchy)
        coarsest_embed_dim = self._h_encoder_dims[-1]
        self.obs_stat_layers.append(self._build_stat_head(self._h_deter_dims[-1] + coarsest_embed_dim, self._h_stoch_dims[-1]))
        self.imgs_stat_layers.append(self._build_stat_head(self._h_deter_dims[-1], self._h_stoch_dims[-1]))
        
        # Finer levels
        for i in range(self._h_levels - 2, -1, -1):
            # Input to stat head: deter_l + embed_l + stoch_{l+1}
            inp_dim = self._h_deter_dims[i] + self._h_encoder_dims[i] + self._h_stoch_dims[i+1]
            self.obs_stat_layers.insert(0, self._build_stat_head(inp_dim, self._h_stoch_dims[i]))
            
            # Input to stat head: deter_l + stoch_{l+1}
            inp_dim_prior = self._h_deter_dims[i] + self._h_stoch_dims[i+1]
            self.imgs_stat_layers.insert(0, self._build_stat_head(inp_dim_prior, self._h_stoch_dims[i]))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )
        
        self.apply(tools.weight_init)
        
        # --------------------------------------------------------------
        #  Build simple 2× up-sampling modules (coarse ➜ fine, L−2 … 0)
        # --------------------------------------------------------------
        self.expand_modules = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)
            for _ in range(self._h_levels - 1)
        ])

        self.apply(tools.weight_init)
        print("hRSSM (with expand) initialised.")

    def spatial_channels(self):
        """
        Calculate the number of channels in the spatial feature map output.
        This is the sum of all hierarchical level features after ladder decoding.
        """
        # Each level contributes (deter_dim + stoch_dim) channels
        total_channels = 0
        for i in range(self._h_levels):
            level_channels = self._h_deter_dims[i] + self._h_stoch_dims[i]
            if self._discrete:
                level_channels = self._h_deter_dims[i] + self._h_stoch_dims[i] * self._discrete
            total_channels += level_channels
        return total_channels

    # ---------- helper: build spatial feature map ---------------------
    def _build_spatial(self, deter_flat, stoch_list):
        """
        Ladder-style top-down decoder that turns {deter, stoch} into a
        spatial tensor ready for the 1×1 RGB head.
        Output shape:  (B, C_final, 2**(L-1), 2**(L-1))
        """
        B = deter_flat.shape[0]
        deter_split = torch.split(deter_flat, self._h_deter_dims, dim=-1)

        # Start from 1×1 coarse map ϕ_L
        feat = torch.cat([deter_split[-1], stoch_list[-1]], -1)       # (B, C_L)
        feat = feat.view(B, -1, 1, 1)                                 # (B, C_L, 1, 1)

        # Walk coarse ➜ fine through expand_modules
        for l in range(self._h_levels - 2, -1, -1):
            feat = self.expand_modules[l](feat)                       # 2× upsample
            add  = torch.cat([deter_split[l], stoch_list[l]], -1)     # inject level-l
            add  = add.view(B, -1, 1, 1).expand_as(feat)
            feat = torch.cat([feat, add], 1)                          # channel-concat
        return feat                                                   # (B, C*, H, W)

    def _build_stat_head(self, inp_dim, stoch_dim):
        layers = [
            nn.Linear(inp_dim, self._h_hidden_dim, bias=False),
            nn.LayerNorm(self._h_hidden_dim, eps=1e-03),
            getattr(torch.nn, "SiLU")()
        ]
        if self._discrete:
            layers.append(nn.Linear(self._h_hidden_dim, stoch_dim * self._discrete))
        else:
            layers.append(nn.Linear(self._h_hidden_dim, 2 * stoch_dim))
        return nn.Sequential(*layers)

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


    # ---------- img_step: now returns (prior, spatial) ----------------
    def img_step(self, prev_state, prev_action, sample=True):
        # -- deterministic update --------------------------------
        prev_stoch_flat = torch.cat(prev_state['stoch'], -1)
        x = torch.cat([prev_stoch_flat, prev_action], -1)
        x = self._img_in_layers(x)
        deter, _ = self._cell(x, [prev_state['deter']])
        deter = deter[0]

        # -- ladder prior ----------------------------------------
        prior_stoch, prior_stats = [], {'mean': [], 'std': [], 'logit': []} \
            if not self._discrete else {'logit': [], 'stoch': []}

        deter_coarse = torch.split(deter, self._h_deter_dims, dim=-1)[-1]
        stats = self._suff_stats_layer_h(self.imgs_stat_layers[-1],
                                         deter_coarse, self._h_stoch_dims[-1])
        z = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
        prior_stoch.append(z)
        for k, v in stats.items(): prior_stats[k].append(v)

        for i in range(self._h_levels - 2, -1, -1):
            deter_l = torch.split(deter, self._h_deter_dims, dim=-1)[i]
            x = torch.cat([deter_l, prior_stoch[-1]], -1)
            stats = self._suff_stats_layer_h(self.imgs_stat_layers[i],
                                             x, self._h_stoch_dims[i])
            z = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
            prior_stoch.insert(0, z)
            for k, v in stats.items(): prior_stats[k].insert(0, v)

        prior = {'stoch': prior_stoch, 'deter': deter, **prior_stats}

        # -------- spatial feature map --------------------------------
        spatial = self._build_spatial(deter, prior_stoch)             # (B, C, H, W)
        return prior, spatial                                         # (prior, spatial)

    # ---------- obs_step: identical but now builds spatial -------------
    def obs_step(self, prev_state, prev_action, embed_list, is_first, sample=True):
        # Handle episode starts
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions), device=self._device)
        elif torch.sum(is_first) > 0:
            # Mask out state and action for new episodes
            is_first = is_first.unsqueeze(-1)
            prev_action *= (1.0 - is_first)
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                if isinstance(val, list):
                    for i in range(len(val)):
                        prev_state[key][i] *= (1.0 - is_first)
                        prev_state[key][i] += (init_state[key][i] * is_first)
                else:
                    prev_state[key] *= (1.0 - is_first)
                    prev_state[key] += (init_state[key] * is_first)
        
        # Get prior
        prior, _ = self.img_step(prev_state, prev_action, sample=sample)
        
        # Compute posterior
        post_stoch = []
        post_stats = {'mean': [], 'std': [], 'logit': []} if not self._discrete else {'logit': [], 'stoch': []}
        
        # Coarsest level
        deter_coarse = torch.split(prior['deter'], self._h_deter_dims, dim=-1)[-1]
        embed_coarse = embed_list[-1].reshape(embed_list[-1].shape[0], -1)
        x = torch.cat([deter_coarse, embed_coarse], -1)
        stats = self._suff_stats_layer_h(self.obs_stat_layers[-1], x, self._h_stoch_dims[-1])
        stoch = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
        post_stoch.append(stoch)
        for k, v in stats.items(): post_stats[k].append(v)
        
        # Finer levels (top-down)
        for i in range(self._h_levels - 2, -1, -1):
            deter_l = torch.split(prior['deter'], self._h_deter_dims, dim=-1)[i]
            embed_l = embed_list[i].reshape(embed_list[i].shape[0], -1)
            # Condition on coarser stochastic state
            x = torch.cat([deter_l, embed_l, post_stoch[-1]], -1)
            stats = self._suff_stats_layer_h(self.obs_stat_layers[i], x, self._h_stoch_dims[i])
            stoch = self.get_dist_h(stats).sample() if sample else self.get_dist_h(stats).mode()
            post_stoch.insert(0, stoch) # Prepend to maintain order
            for k, v in stats.items(): post_stats[k].insert(0, v)

        post = {'stoch': post_stoch, 'deter': prior['deter'], **post_stats}
        spatial_post = self._build_spatial(prior['deter'], post_stoch) 
        return post, prior, spatial_post

    def observe(self, embed_list, action, is_first, state=None):
        # time-major swap for each level
        swap = lambda x: x.permute([1, 0] + list(range(2, x.ndim)))
        embed_list_T = [swap(e) for e in embed_list]
        action_T, is_first_T = swap(action), swap(is_first)

        def step(prev, a, embeds, first):
            post, spatial = self.obs_step(prev[0], a, embeds, first, sample=True)
            # we also need the prior for KL; recompute it deterministically from the same prev state & action
            prior, _ = self.img_step(prev[0], a, sample=True)
            return (post, prior, spatial), (post, prior)

        (post_seq, prior_seq, spatial_seq), _ = tools.static_scan(
            step, (action_T, list(zip(*embed_list_T)), is_first_T), (state, state)
        )

        unswap = lambda t: t.permute([1, 0] + list(range(2, t.ndim)))
        # batch-major
        post_seq  = {k: (v if not isinstance(v, list) else [unswap(x) for x in v]) for k, v in post_seq.items()}
        prior_seq = {k: (v if not isinstance(v, list) else [unswap(x) for x in v]) for k, v in prior_seq.items()}
        spatial_seq = unswap(spatial_seq)  # (B,T,C,H,W)
        return post_seq, prior_seq, spatial_seq
    
    # ------------------------------------------------------------------
    #   Imagination rollout (open-loop): given an initial posterior
    #   `state` and a sequence of future actions, predict priors and
    #   their spatial maps for every step.
    #   Returns:
    #       prior_seq   – dict with same keys as a single prior, each
    #                     value shaped (B, T, …)
    #       spatial_seq – tensor (B, T, C, H, W)
    # ------------------------------------------------------------------
    def imagine_with_action(self, action, state, sample=True):
        """
        action : (B, T, act_dim)
        state  : hierarchical posterior at time 0  (B, ...)
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, x.ndim)))   # B,T → T,B
        action_T = swap(action)                                       # (T,B,act)

        # one scan step ----------------------------------------------------------------
        def step(prev_state, act_t):
            prior_t, spatial_t = self.img_step(prev_state, act_t, sample=sample)
            return prior_t, (prior_t, spatial_t)

        # run the GRU / ladder forward --------------------------------------------------
        (priors_T, spatials_T), _ = tools.static_scan(
            step, [action_T], state
        )  # priors_T is a dict with time major tensors; spatials_T is (T,B,C,H,W)

        # time-major → batch-major ------------------------------------------------------
        unswap = lambda t: t.permute([1, 0] + list(range(2, t.ndim)))  # T,B → B,T
        prior_seq = {
            k: (v if not isinstance(v, list)
                else [unswap(x) for x in v])        # lists (one per level)
            for k, v in priors_T.items()
        }
        spatial_seq = unswap(spatials_T)            # (B,T,C,H,W)

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
        
        dyn_loss = torch.stack(dyn_loss_list).sum(0)
        rep_loss = torch.stack(rep_loss_list).sum(0)
        kl_value = torch.stack(kl_value_list).sum(0)

        dyn_loss = torch.clip(dyn_loss, min=free)
        rep_loss = torch.clip(rep_loss, min=free)
        
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, kl_value, dyn_loss, rep_loss

    def get_feat(self, state, levels='all'):
        if levels == 'all':
            stoch_flat = torch.cat(state['stoch'], -1)
            return torch.cat([stoch_flat, state['deter']], -1)
        elif levels == 'top':
            return torch.cat([state['stoch'][-1], torch.split(state['deter'], self._h_deter_dims, dim=-1)[-1]], -1)
        else: # Assume integer index
            level_idx = int(levels)
            return torch.cat([state['stoch'][level_idx], torch.split(state['deter'], self._h_deter_dims, dim=-1)[level_idx]], -1)
    
    def get_dist_h(self, state_level):
        if self._discrete:
            logit = state_level["logit"]
            return tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio)
        else:
            mean, std = state_level["mean"], state_level["std"]
            return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
    
    def get_dist(self, full_state):
        """
        full_state is the usual dict {'stoch':[z1,…,zL], 'mean':[…], …}
        Returns a _ProductDist so existing Dreamer code keeps working.
        """
        dists = [self.get_dist_h({k: (v[i] if isinstance(v, list) else v)
                                for k, v in full_state.items()})
                for i in range(self._h_levels)]
        return _ProductDist(dists)

    def _suff_stats_layer_h(self, layer, x, stoch_dim):
        stats = layer(x)
        if self._discrete:
            logit = stats.reshape(list(x.shape[:-1]) + [stoch_dim, self._discrete])
            return {"logit": logit}
        else:
            mean, std = torch.split(stats, [stoch_dim] * 2, -1)
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
