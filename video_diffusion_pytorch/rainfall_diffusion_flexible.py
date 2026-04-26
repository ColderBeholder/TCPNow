import math
import copy

import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T
from torch.cuda.amp import autocast, GradScaler
try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

import torchvision.models as models

from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM
from video_diffusion_pytorch.Env_transformer import Env_net

import logging


# helpers functions

def exists(x):
    return x is not None

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])



def _require_even_hw(x):
    h, w = x.shape[-2], x.shape[-1]
    if (h % 2) != 0 or (w % 2) != 0:
        raise ValueError(f"Wavelet-domain diffusion requires even H and W, got {(h, w)}")


def haar_dwt2d_tiled(x: torch.Tensor) -> torch.Tensor:
    """Single-level orthonormal Haar DWT packed back into the original HxW grid.

    Input / output shape: (..., H, W) -> (..., H, W)
    Quadrants in the output are [LL | LH; HL | HH].
    """
    _require_even_hw(x)
    x00 = x[..., 0::2, 0::2]
    x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]
    x11 = x[..., 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    top = torch.cat([ll, lh], dim=-1)
    bottom = torch.cat([hl, hh], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def haar_idwt2d_tiled(x: torch.Tensor) -> torch.Tensor:
    """Inverse of haar_dwt2d_tiled with the same packed HxW layout."""
    _require_even_hw(x)
    h2, w2 = x.shape[-2] // 2, x.shape[-1] // 2
    ll = x[..., :h2, :w2]
    lh = x[..., :h2, w2:]
    hl = x[..., h2:, :w2]
    hh = x[..., h2:, w2:]

    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5

    out = torch.empty_like(x)
    out[..., 0::2, 0::2] = x00
    out[..., 0::2, 1::2] = x01
    out[..., 1::2, 0::2] = x10
    out[..., 1::2, 1::2] = x11
    return out


def wavelet_detail_mask_like(x: torch.Tensor) -> torch.Tensor:
    """Mask of the packed high-frequency quadrants [LH, HL, HH]."""
    _require_even_hw(x)
    mask = torch.zeros_like(x)
    h2, w2 = x.shape[-2] // 2, x.shape[-1] // 2
    mask[..., :h2, w2:] = 1.
    mask[..., h2:, :w2] = 1.
    mask[..., h2:, w2:] = 1.
    return mask

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class StandardSpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h = self.heads)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

class GatedSpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias = False)
        self.to_gate = nn.Conv2d(dim, hidden_dim, 1, bias = True)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        g = self.to_gate(x)
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q, k, v, g = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads),
            (q, k, v, g)
        )

        # positive features for linear attention
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        g = torch.sigmoid(g)

        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        k_sum = k.sum(dim = 2)
        z = 1.0 / torch.einsum('b h n d, b h d -> b h n', q, k_sum).clamp(min = 1e-6)
        out = torch.einsum('b h n d, b h d e -> b h n e', q * self.scale, kv)
        out = out * z.unsqueeze(-1)
        out = out * g

        out = rearrange(out, 'b h (x y) c -> b (h c) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

class RuntimeSpatialAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, use_gla = True):
        super().__init__()
        self.use_gla = use_gla
        self.standard = StandardSpatialLinearAttention(dim, heads = heads, dim_head = dim_head)
        self.gated = GatedSpatialLinearAttention(dim, heads = heads, dim_head = dim_head)

    def set_use_gla(self, use_gla):
        self.use_gla = use_gla

    def forward(self, x):
        if self.use_gla:
            return self.gated(x)
        return self.standard(x)

# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x

class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_heads = 8,
        attn_dim_head = 32,
        use_bert_text_cond = False,
        init_dim = None,
        init_kernel_size = 7,
        use_sparse_linear_attn = True,
        use_gla = True,
        block_type = 'resnet',
        resnet_groups = 8,
        output_frames=4,
        ifs_channels=11,
        ifs_out_channel=None,
        multi_sc = ''
    ):
        super().__init__()
        self.channels = channels
        self.use_gla = use_gla

        # temporal attention and its relative positional encoding
        self.output_frames = output_frames
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom(
            'b c f h w',
            'b (h w) f c',
            RuntimeAttention(
                dim,
                heads = attn_heads,
                dim_head = attn_dim_head,
                rotary_emb = rotary_emb,
                use_gla = self.use_gla
            )
        )

        # self.time_rel_pos_bias = RelativePositionBias(heads = attn_heads, max_distance = 32) # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (1, init_kernel_size, init_kernel_size), padding = (0, init_padding, init_padding))

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning


        self.has_cond = exists(cond_dim) or use_bert_text_cond
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else cond_dim
        self.cond_dim = cond_dim
        self.cond_encoder = Env_net(env_list=multi_sc).cuda()

        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None

        cond_dim = time_dim + int(cond_dim or 0)+512 # 512: future dim

        # ifs conditioning
        self.ifs_channels = ifs_channels
        self.ifs_encoder = ifs_encoder(in_channels=ifs_channels)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # block type
        block_klass = partial(ResnetBlock, groups = resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim = cond_dim)

        # modules for all layers
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, RuntimeSpatialAttention(dim_out, heads = attn_heads, use_gla = self.use_gla))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom(
            'b c f h w',
            'b f (h w) c',
            RuntimeAttention(mid_dim, heads = attn_heads, use_gla = self.use_gla)
        )

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                Residual(PreNorm(dim_in, RuntimeSpatialAttention(dim_in, heads = attn_heads, use_gla = self.use_gla))) if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def set_runtime_features(self, use_gla = None):
        if use_gla is None:
            return
        self.use_gla = use_gla
        for module in self.modules():
            if hasattr(module, 'set_use_gla'):
                module.set_use_gla(use_gla)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x, #data_obs_diff, data_obs_real, ERA5_obs, ERA5_ifs, noise
        time,
        cond = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond) and self.cond_encoder is None), \
            'cond must be passed in if cond_dim specified and no cond_encoder is provided'
        batch, device = x.shape[0], x.device

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        assert self.channels + self.ifs_channels == x.shape[1]
        x_ifs = x[:, self.channels - 1:-1]
        x = torch.cat([x[:, :self.channels - 1], x[:, -1:]], dim=1)

        x_ifs_feature = self.ifs_encoder(x_ifs)

        x = self.init_conv(x)

        x = self.init_temporal_attn(x) # x = self.init_temporal_attn(x, pos_bias = time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            cond_features = cond
            if self.cond_encoder is not None:
                assert cond is not None, 'environmental condition inputs are required when cond_encoder is set'
                if torch.is_tensor(cond):
                    cond_features = cond
                else:
                    cond_features = self.cond_encoder(cond)
            if not torch.is_tensor(cond_features):
                raise TypeError('cond must be a tensor once provided or encoded inside Unet3D')
            cond_features = cond_features.to(device)
            mask = prob_mask_like((batch,), null_cond_prob, device = device)
            cond_features = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond_features)
            t = torch.cat((t, cond_features,x_ifs_feature), dim = -1)

        h = []

        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, focus_present_mask = focus_present_mask) # x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, focus_present_mask = focus_present_mask) # x = self.mid_temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = spatial_attn(x)
            x = temporal_attn(x, focus_present_mask = focus_present_mask) # x = temporal_attn(x, pos_bias = time_rel_pos_bias, focus_present_mask = focus_present_mask)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        return self.final_conv(x)

class StandardAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)
        q = q * self.scale

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

class GatedAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_k = nn.Linear(dim, hidden_dim, bias = False)
        self.to_v = nn.Linear(dim, hidden_dim, bias = False)
        self.to_gate = nn.Linear(dim, hidden_dim, bias = True)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
        self,
        x,
        pos_bias = None,
        focus_present_mask = None
    ):
        n = x.shape[-2]

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        g = self.to_gate(x)

        q, k, v, g = rearrange_many((q, k, v, g), '... n (h d) -> ... h n d', h=self.heads)

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        g = torch.sigmoid(g)

        if exists(focus_present_mask):
            fp = rearrange(focus_present_mask.float(), 'b -> b 1 1 1 1')
            g = g * (1.0 - 0.5 * fp)

        kv = einsum('... h n d, ... h n e -> ... h d e', k, v)
        k_sum = k.sum(dim=-2)
        z = 1.0 / einsum('... h n d, ... h d -> ... h n', q, k_sum).clamp(min=1e-6)
        out = einsum('... h n d, ... h d e -> ... h n e', q * self.scale, kv)
        out = out * z.unsqueeze(-1)
        out = out * g

        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

class RuntimeAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        rotary_emb = None,
        use_gla = True
    ):
        super().__init__()
        self.use_gla = use_gla
        self.standard = StandardAttention(
            dim,
            heads = heads,
            dim_head = dim_head,
            rotary_emb = rotary_emb
        )
        self.gated = GatedAttention(
            dim,
            heads = heads,
            dim_head = dim_head,
            rotary_emb = rotary_emb
        )

    def set_use_gla(self, use_gla):
        self.use_gla = use_gla

    def forward(self, x, **kwargs):
        if self.use_gla:
            return self.gated(x, **kwargs)
        return self.standard(x, **kwargs)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l2',
        use_dynamic_thres = False,
        dynamic_thres_percentile = 0.9,
        input_frames = 4,
        output_frames = 4,
        obs_channels = 1,
        use_gla = None,
        use_wavelet_domain = False,
        wavelet_detail_weight = 2.0,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.denoise_fn = denoise_fn
        self.use_gla = use_gla
        self.use_wavelet_domain = use_wavelet_domain
        self.wavelet_detail_weight = wavelet_detail_weight
        if exists(self.use_gla) and hasattr(self.denoise_fn, 'set_runtime_features'):
            self.denoise_fn.set_runtime_features(use_gla = self.use_gla)

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def to_wavelet_domain(self, x):
        if not self.use_wavelet_domain:
            return x
        b, c, f, h, w = x.shape
        x_reshaped = rearrange(x, 'b c f h w -> (b c f) h w')
        x_wave = haar_dwt2d_tiled(x_reshaped)
        return rearrange(x_wave, '(b c f) h w -> b c f h w', b=b, c=c, f=f)

    def from_wavelet_domain(self, x):
        if not self.use_wavelet_domain:
            return x
        b, c, f, h, w = x.shape
        x_reshaped = rearrange(x, 'b c f h w -> (b c f) h w')
        x_img = haar_idwt2d_tiled(x_reshaped)
        return rearrange(x_img, '(b c f) h w -> b c f h w', b=b, c=c, f=f)

    def wavelet_detail_loss(self, target, pred):
        if not self.use_wavelet_domain or self.wavelet_detail_weight <= 0:
            return target.new_tensor(0.)
        mask = wavelet_detail_mask_like(target)
        if self.loss_type == 'l1':
            return F.l1_loss(pred * mask, target * mask)
        return F.mse_loss(pred * mask, target * mask)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        x_t = x[:,-1:, -self.output_frames:]
        x_recon = self.predict_start_from_noise(x_t, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        x_recon = x_recon[:,:, -self.output_frames:]

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x[:,-1:,-self.output_frames:])
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.,step_img=None,obs_data=None):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        img_list = []
        # self.num_timesteps = 1
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # print(obs_data.shape,img.shape)
            img = torch.cat([obs_data, img], dim=1)
            # print(img.shape)
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                cond = cond, cond_scale = cond_scale)
            if step_img is not None:
                step_real = int(step_img*self.num_timesteps)
                if i % step_real ==0:
                    # print(i)
                    img_list.append(img)

        img = self.from_wavelet_domain(img)
        if step_img is not None:
            img_steps = torch.stack(img_list, dim=1)
            bs, steps, c, f, h, w = img_steps.shape
            img_steps = rearrange(img_steps, 'b s c f h w -> (b s) c f h w')
            img_steps = self.from_wavelet_domain(img_steps)
            img_steps = rearrange(img_steps, '(b s) c f h w -> b s c f h w', b=bs, s=steps)
            return unnormalize_img(img), unnormalize_img(img_steps)
        else:
            return unnormalize_img(img)

    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16,data_condition=None,step_img=None,obs_data=None):
        device = next(self.denoise_fn.parameters()).device

        cond_inputs = cond
        if is_list_str(cond_inputs):
            cond_inputs = bert_embed(tokenize(cond_inputs)).to(device)
        if cond_inputs is None and data_condition is not None:
            cond_inputs = data_condition

        batch_size = cond_inputs.shape[0] if (isinstance(cond_inputs, torch.Tensor) and exists(cond_inputs)) else obs_data.shape[0]
        image_size = self.image_size
        channels = 1
        num_frames = self.output_frames

        # please keep the input same with the training!!!!!!!!!!!!!!!!!!!!!!!!!
        obs_data = normalize_img(obs_data)
        obs_data = self.to_wavelet_domain(obs_data)
        # obs_data = self.init_conv(obs_data)

        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size),
                                  cond = cond_inputs, cond_scale = cond_scale,
                                  step_img=step_img,obs_data=obs_data)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start,obs_data, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device

        x_start = self.to_wavelet_domain(x_start)
        obs_data = self.to_wavelet_domain(obs_data)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        unet_input = torch.cat([obs_data,x_noisy],dim=1)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond))
            cond = cond.to(device)

        # just get num_pre
        x_recon = self.denoise_fn(unet_input, t, cond = cond, **kwargs)[:,:,-f:]
        # print(x_recon.shape)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        loss = loss + self.wavelet_detail_weight * self.wavelet_detail_loss(noise, x_recon)
        return loss

    def forward(self, x,obs_data,data_condition, *args, **kwargs):
        #x --->  b c f h w
        b, device, img_size, = x.shape[0], x.device, self.image_size
        # print(x.shape)
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames-obs_data.shape[2], h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # Normal
        x = normalize_img(x)
        obs_data = normalize_img(obs_data)
        # b c-allera5+2 f h w
        # obs_data = self.init_conv(obs_data)
        ##########################################
        return self.p_losses(x, obs_data, t,cond=data_condition, *args, **kwargs)

# trainer class

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

# trainer class

class ifs_encoder(torch.nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(ifs_encoder, self).__init__()
        self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = torch.nn.Conv2d(in_channels*4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        new_fc = nn.Linear(self.resnet.fc.in_features,512)
        self.resnet.fc = new_fc

    def forward(self, x):
        x = x*2 -1
        x = rearrange(x, 'b c f h w -> b (c f) h w')
        ifs_feature = self.resnet(x)
        return ifs_feature

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        ema_decay = 0.995,
        num_frames = 16,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 10,
        results_folder = './results',
        num_sample_rows = 4,
        max_grad_norm = None,
        dataset_train=None,
        dataset_val=None,
        use_accelerate=False,
        accelerator=None,
        accelerator_kwargs=None,
        pre_milestone = 0,
    ):
        super().__init__()
        self.accelerator = accelerator
        self.use_accelerate = (accelerator is not None) or use_accelerate
        if self.use_accelerate and self.accelerator is None:
            if Accelerator is None:
                raise ImportError('Hugging Face accelerate is required when use_accelerate=True')
            accel_kwargs = dict(accelerator_kwargs or {})
            accel_kwargs.setdefault('gradient_accumulation_steps', gradient_accumulate_every)
            accel_kwargs.setdefault('mixed_precision', 'fp16' if amp else 'no')
            self.accelerator = Accelerator(**accel_kwargs)
        self.is_main_process = self.accelerator.is_main_process if self.use_accelerate else True
        device = self.accelerator.device if self.use_accelerate else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model = diffusion_model.to(device)
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model).to(device)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every

        num_processes = 1
        if self.use_accelerate:
            num_processes = max(1, getattr(self.accelerator.state, 'num_processes', getattr(self.accelerator, 'num_processes', 1)))
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            num_processes = torch.distributed.get_world_size()
        self.num_processes = num_processes
        if num_processes > 1:
            # split the global requested steps evenly across processes
            self.train_num_steps = max(1, math.ceil(train_num_steps / num_processes))
        else:
            self.train_num_steps = train_num_steps

        self.ds = dataset_train
        train_loader = None
        if self.ds is not None:
            assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'
            train_loader = data.DataLoader(
                self.ds,
                batch_size = train_batch_size,
                shuffle = True,
                pin_memory = True,
                collate_fn = self.ds.collate_data
            )

        self.dl_val = None
        if dataset_val is not None:
            self.dl_val = data.DataLoader(
                dataset_val,
                batch_size = train_batch_size,
                shuffle = False,
                pin_memory = True,
                collate_fn = dataset_val.collate_data
            )

        self.opt = Adam(self.model.parameters(), lr = train_lr)

        if self.use_accelerate:
            to_prepare = [self.model, self.opt]
            if train_loader is not None:
                to_prepare.append(train_loader)
            if self.dl_val is not None:
                to_prepare.append(self.dl_val)
            prepared = self.accelerator.prepare(*to_prepare)
            self.model, self.opt = prepared[:2]
            idx = 2
            if train_loader is not None:
                train_loader = prepared[idx]
                idx += 1
            if self.dl_val is not None:
                self.dl_val = prepared[idx]

        if train_loader is not None:
            self.dl = cycle(train_loader)

        self.step = 0

        self.amp = amp
        self.scaler = None if self.use_accelerate else GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

        if pre_milestone:
            self.load(pre_milestone)

    def _trainer_model(self):
        return self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model

    def _autocast_context(self):
        return self.accelerator.autocast() if self.use_accelerate else autocast(enabled = self.amp)

    def _prepare_batch(self, batch):
        device = self.device

        def to_device(tensor):
            return tensor.to(device, non_blocking = True)

        data_obs = to_device(batch['obs_rain'])
        data_obs_diff = to_device(batch['obs_diff'])
        era_obs = to_device(batch['modal_env']['obs'])
        era_pre = to_device(batch['modal_env']['pre'])

        env_obs = {key: to_device(val) for key, val in batch['env_data'].items()}
        obs_stack = torch.cat([data_obs_diff, data_obs, era_obs, era_pre], dim = 1)
        data_target_diff = to_device(batch['pre_diff'])
        return data_target_diff, obs_stack, env_obs

    def _forward_loss(self, batch, prob_focus_present, focus_present_mask):
        data_target_diff, obs_stack, data_condition = self._prepare_batch(batch)
        mask = focus_present_mask
        if isinstance(mask, torch.Tensor):
            mask = mask.to(self.device)

        with self._autocast_context():
            loss = self.model(
                data_target_diff,
                obs_data = obs_stack,
                data_condition = data_condition,
                prob_focus_present = prob_focus_present,
                focus_present_mask = mask
            )
        return loss

    def _loss_value(self, loss):
        if self.use_accelerate:
            gathered = self.accelerator.gather(loss.detach())
            return gathered.mean().item()
        return loss.detach().float().item()

    def _after_optimization(self):
        if self.step % self.update_ema_every == 0:
            self.step_ema()
        self.step += 1

    def _train_iteration(self, prob_focus_present, focus_present_mask):
        if self.use_accelerate:
            batch = next(self.dl)
            with self.accelerator.accumulate(self.model):
                loss = self._forward_loss(batch, prob_focus_present, focus_present_mask)
                self.accelerator.backward(loss)
                sync_gradients = self.accelerator.sync_gradients
                if exists(self.max_grad_norm) and sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
            if sync_gradients:
                self._after_optimization()
            return self._loss_value(loss), sync_gradients

        loss = None
        for _ in range(self.gradient_accumulate_every):
            batch = next(self.dl)
            loss = self._forward_loss(batch, prob_focus_present, focus_present_mask)
            self.scaler.scale(loss / self.gradient_accumulate_every).backward()

        if exists(self.max_grad_norm):
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad()
        self._after_optimization()
        return self._loss_value(loss), True

    def reset_parameters(self):
        self.ema_model.load_state_dict(self._trainer_model().state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self._trainer_model())

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self._trainer_model().state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
            'opt': self.opt.state_dict(),
        }
        if self.use_accelerate:
            self.accelerator.save(data, str(self.results_folder / f'model-{milestone}.pt'))
            return
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location = self.device)
            print(f'loading the check point model-{milestone}.pt')

            self.step = data['step']
            self._trainer_model().load_state_dict(data['model'], **kwargs)
            self.ema_model.load_state_dict(data['ema'], **kwargs)
            self.opt.load_state_dict(data['opt'])
            if self.scaler is not None and data.get('scaler') is not None:
                self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler('./Exps/log/log.log'),
            ]
            )
        if not hasattr(self, 'dl'):
            raise ValueError('Training dataset must be provided before calling train()')

        progress_bar = tqdm(total = self.train_num_steps, initial = self.step, disable = not self.is_main_process)
        while self.step < self.train_num_steps:
            loss_value, performed_update = self._train_iteration(prob_focus_present, focus_present_mask)
            if performed_update and self.is_main_process:
                progress_bar.set_description(f'{self.step}: {loss_value}')
                progress_bar.update(1)
                if self.step%20==0:
                    logging.info(f'Step {self.step}: loss {loss_value}')
                if self.step%1e4==0:
                    self.save(self.step)

        progress_bar.close()
        if self.is_main_process:
            print('training completed')
