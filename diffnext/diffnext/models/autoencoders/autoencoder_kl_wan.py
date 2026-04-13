# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
"""Simple implementation of AutoEncoderKL for Wan."""

from einops import rearrange
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin


class Conv3d(nn.Conv3d):
    """3D convolution layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = [self.padding[2]] * 2 + [self.padding[1]] * 2
        self._padding = tuple(self._padding + [2 * self.padding[0], 0])
        self.padding = (0, 0, 0)

    def forward_cache(self, x, feat_cache, feat_idx) -> torch.Tensor:
        idx, feat_idx[0] = feat_idx[0], feat_idx[0] + 1
        cache_x = x[:, :, -2:].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat([feat_cache[idx][:, :, -1:], cache_x], dim=2)
        x, feat_cache[idx] = self(x, feat_cache[idx]), cache_x
        return x

    def forward(self, x, cache_x=None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            padding[4] -= cache_x.shape[2]
            x = torch.cat([cache_x.to(x.device), x], dim=2)
        return super().forward(nn.functional.pad(x, padding))


class RMSNorm(nn.Module):
    """RMS normalization layer."""

    def __init__(self, dim, images=False):
        super().__init__()
        broadcastable_dims = (1, 1) if images else (1, 1, 1)
        self.gamma = nn.Parameter(torch.ones((dim, *broadcastable_dims)))

    def forward(self, x) -> torch.Tensor:
        scale = self.gamma.mul(self.gamma.numel() ** 0.5)
        return nn.functional.normalize(x, dim=1) * scale


class AvgDown3D(nn.Module):
    """Average downsample layer."""

    def __init__(self, in_dim, out_dim, factor_t, factor_s=1):
        super().__init__()
        self.out_dim, self.factor_t = out_dim, factor_t
        self.patch_args = {"r": factor_t, "p": factor_s, "q": factor_s}
        self.patch_args["pattern"] = "b c (t r) (h p) (w q) -> b (c r p q) t h w"

    def forward(self, x) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = nn.functional.pad(x, (0, 0, 0, 0, pad_t, 0)) if pad_t else x
        x = rearrange(x, **self.patch_args)
        return x.unflatten(1, (self.out_dim, -1)).mean(dim=2)


class DupUp3D(nn.Module):
    """Duplicate upsample layer."""

    def __init__(self, in_dim, out_dim, factor_t, factor_s=2):
        super().__init__()
        self.first_chunk = True
        self.repeats = out_dim * factor_t * factor_s**2 // in_dim
        self.patch_args = {"r": factor_t, "p": factor_s, "q": factor_s}
        self.patch_args["pattern"] = "b (c r p q) t h w -> b c (t r) (h p) (w q)"

    def forward(self, x) -> torch.Tensor:
        x = rearrange(x.repeat_interleave(self.repeats, dim=1), **self.patch_args)
        return x[:, :, -1:] if self.first_chunk else x


class Resample(nn.Module):
    """Resample layer"""

    def __init__(self, dim, mode, upsample_out_dim=None):
        super().__init__()
        self.mode = mode
        if self.mode in ["upsample2d", "upsample3d"]:
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode="nearest-exact"))
            self.resample.append(nn.Conv2d(dim, upsample_out_dim or dim // 2, 3, padding=1))
        if self.mode in ["downsample2d", "downsample3d"]:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1) * 2), nn.Conv2d(dim, dim, 3, 2))
        if mode in ["upsample3d"]:
            self.time_conv = Conv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        if mode in ["downsample3d"]:
            self.time_conv = Conv3d(dim, dim, (3, 1, 1), (2, 1, 1), padding=(0, 0, 0))

    def forward_downsample3d(self, x, feat_cache=None, feat_idx=[0]) -> torch.Tensor:
        if self.mode != "downsample3d" or feat_cache is None:
            return x
        idx, feat_idx[0] = feat_idx[0], feat_idx[0] + 1
        if feat_cache[idx] is None:
            feat_cache[idx] = x[:, :, -1:].clone()
            return x
        cache_x, feat_cache[idx] = feat_cache[idx][:, :, -1:], x[:, :, -1:].clone()
        return self.time_conv(torch.cat([cache_x, x], 2))

    def forward_upsample3d(self, x, feat_cache=None, feat_idx=[0]) -> torch.Tensor:
        if self.mode != "upsample3d" or feat_cache is None:
            return x
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx], feat_idx[0] = torch.zeros_like(x), feat_idx[0] + 1
            return x
        x = self.time_conv.forward_cache(x, feat_cache, feat_idx)
        return rearrange(x, "b (d c) t h w -> b c (t d) h w", d=2)

    def forward(self, x, feat_cache=None, feat_idx=[0]) -> torch.Tensor:
        x = self.forward_upsample3d(x, feat_cache, feat_idx)
        x, t = x.transpose(1, 2).flatten(0, 1), x.shape[2]  # 3D -> 2D
        x = self.resample(x).unflatten(0, (-1, t)).transpose(1, 2)  # 2D -> 3D
        return self.forward_downsample3d(x, feat_cache, feat_idx)


class Attention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm = RMSNorm(dim, images=True)

    def forward(self, x) -> torch.Tensor:
        shortcut, (time, _, width) = x, x.shape[-3:]
        x = self.norm(x.transpose(1, 2).flatten(0, 1))
        q, k, v = self.to_qkv(x).flatten(2, 3).unsqueeze(1).transpose(2, 3).chunk(3, -1)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = self.proj(x.squeeze(1).transpose(1, 2).unflatten(-1, (-1, width)))
        x = x.unflatten(0, (-1, time)).transpose(1, 2)
        return x.add_(shortcut)


class WanResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.norm1, self.norm2 = RMSNorm(in_dim), RMSNorm(out_dim)
        self.conv1 = Conv3d(in_dim, out_dim, 3, padding=1)
        self.conv2 = Conv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = Conv3d(in_dim, out_dim, 1) if in_dim != out_dim else None
        self.nonlinearity, self.dropout = nn.SiLU(), nn.Dropout(0)

    def forward(self, x, feat_cache=None, feat_idx=[0]) -> torch.Tensor:
        shortcut = self.conv_shortcut(x) if self.conv_shortcut else x
        x = self.nonlinearity(self.norm1(x))
        x = self.conv1.forward_cache(x, feat_cache, feat_idx) if feat_cache else self.conv1(x)
        x = self.dropout(self.nonlinearity(self.norm2(x)))
        x = self.conv2.forward_cache(x, feat_cache, feat_idx) if feat_cache else self.conv2(x)
        return x.add_(shortcut)


class WanMidBlock(nn.Module):
    """Middle block."""

    def __init__(self, dim, num_layers=1):
        super().__init__()
        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList([WanResBlock(dim, dim)])
        for _ in range(num_layers):
            self.attentions.append(Attention(dim))
            self.resnets.append(WanResBlock(dim, dim))

    def forward(self, x, feat_cache=None, feat_idx=[0]) -> torch.Tensor:
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x), feat_cache, feat_idx)
        return x


class WanDownBlock(nn.Module):
    """Downsample block."""

    def __init__(self, in_dim, out_dim, num_blocks, temperal=False, down=False):
        super().__init__()
        mode = "downsample{}d".format(3 if temperal else 2)
        factor_t, factor_s = 2 if temperal else 1, 2 if down else 1
        self.shortcut = AvgDown3D(in_dim, out_dim, factor_t, factor_s)
        self.resnets = nn.ModuleList()
        for i in range(num_blocks):
            self.resnets.append(WanResBlock(out_dim if i else in_dim, out_dim))
        self.downsampler = Resample(out_dim, mode) if down else None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        shortcut = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        x = self.downsampler(x, feat_cache, feat_idx) if self.downsampler else x
        return x.add_(self.shortcut(shortcut))


class WanUpBlock(nn.Module):
    """Upsample block."""

    def __init__(self, in_dim, out_dim, num_blocks, temperal=False, up=False, residual=True):
        super().__init__()
        mode = "upsample{}d".format(3 if temperal else 2)
        self.resnets = nn.ModuleList()
        for i in range(num_blocks + 1):
            self.resnets.append(WanResBlock(out_dim if i else in_dim, out_dim))
        self.upsampler = Resample(out_dim, mode, out_dim) if up and residual else None
        self.upsamplers = nn.ModuleList([Resample(out_dim, mode)]) if up and not residual else None
        self.shortcut = DupUp3D(in_dim, out_dim, 2 if temperal else 1) if residual and up else None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        shortcut = x if self.shortcut else None
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        x = self.upsampler(x, feat_cache, feat_idx) if self.upsampler else x
        x = self.upsamplers[0](x, feat_cache, feat_idx) if self.upsamplers else x
        return x.add_(self.shortcut(shortcut)) if self.shortcut else x


class WanEncoder3d(nn.Module):
    """VAE encoder."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal=[True, True, False],
        is_residual=False,  # Residual for Wan 2.2
        in_channels=3,
    ):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = Conv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(
                    WanDownBlock(
                        in_dim,
                        out_dim,
                        num_res_blocks,
                        temperal=temperal[i] if i != len(dim_mult) - 1 else False,
                        down=i != len(dim_mult) - 1,
                    )
                )
            else:
                for j in range(num_res_blocks):
                    self.down_blocks.append(WanResBlock(out_dim if j else in_dim, out_dim))
                if i != len(dim_mult) - 1:
                    resample = "downsample{}d".format(3 if temperal[i] else 2)
                    self.down_blocks.append(Resample(out_dim, resample))
        self.mid_block = WanMidBlock(out_dim)
        self.norm_out, self.nonlinearity = RMSNorm(out_dim), nn.SiLU()
        self.conv_out = Conv3d(out_dim, z_dim * 2, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.conv_in.forward_cache(x, feat_cache, feat_idx) if feat_cache else self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        x = self.nonlinearity(self.norm_out(x))
        x = self.conv_out.forward_cache(x, feat_cache, feat_idx) if feat_cache else self.conv_out(x)
        return x


class WanDecoder3d(nn.Module):
    """VAE decoder."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal=[False, True, True],
        is_residual=False,  # Residual for Wan 2.2
        out_channels=3,
    ):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = Conv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0])
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            self.up_blocks.append(
                WanUpBlock(
                    in_dim if is_residual or i == 0 else in_dim // 2,
                    out_dim,
                    num_res_blocks,
                    temperal=temperal[i] if i != len(dim_mult) - 1 else False,
                    up=i != len(dim_mult) - 1,
                    residual=is_residual,
                )
            )
        self.norm_out, self.nonlinearity = RMSNorm(out_dim), nn.SiLU()
        self.conv_out = Conv3d(out_dim, out_channels, 3, padding=1)
        self.first_chunk_modules = [m for m in self.modules() if hasattr(m, "first_chunk")]

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        [setattr(m, "first_chunk", first_chunk) for m in self.first_chunk_modules]
        x = self.conv_in.forward_cache(x, feat_cache, feat_idx) if feat_cache else self.conv_in(x)
        x = self.mid_block(x, feat_cache, feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)
        x = self.nonlinearity(self.norm_out(x))
        x = self.conv_out.forward_cache(x, feat_cache, feat_idx) if feat_cache else self.conv_out(x)
        return x


class AutoencoderKLWan(ModelMixin, AutoencoderMixin, ConfigMixin):
    """AutoEncoder KL."""

    @register_to_config
    def __init__(
        self,
        base_dim=160,
        decoder_base_dim=256,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[False, True, True],
        is_residual=False,
        in_channels=12,
        out_channels=12,
        latents_mean=[],
        latents_std=[],
        attn_scales=[],
        dropout=0.0,
        patch_size=2,
        scale_factor_temporal=4,
        scale_factor_spatial=16,
    ) -> None:
        super().__init__()
        self.encoder = WanEncoder3d(
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal=temperal_downsample,
            is_residual=is_residual,
            in_channels=in_channels,
        )
        self.decoder = WanDecoder3d(
            dim=decoder_base_dim or base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal=temperal_downsample[::-1],
            is_residual=is_residual,
            out_channels=out_channels,
        )
        self.quant_conv = Conv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = Conv3d(z_dim, z_dim, 1)
        self.register_buffer("shift_factors", torch.as_tensor(latents_mean), persistent=False)
        self.register_buffer("scaling_factors", torch.as_tensor(latents_std), persistent=False)
        self.latent_dist = DiagonalGaussianDistribution
        self.feat_map = [None] * sum(isinstance(m, Conv3d) for m in self.decoder.modules())
        self.enc_feat_map = [None] * sum(isinstance(m, Conv3d) for m in self.encoder.modules())

    def scale_(self, x) -> torch.Tensor:
        """Scale the input latents."""
        return x.sub_(self.shift_factors).mul_(self.scaling_factors)

    def unscale_(self, x) -> torch.Tensor:
        """Unscale the input latents."""
        return x.div_(self.scaling_factors).add_(self.shift_factors)

    def clear_cache(self):
        """Clear feature cache."""
        self.feat_map = [None] * len(self.feat_map)
        self.enc_feat_map = [None] * len(self.enc_feat_map)

    def encode(self, x) -> AutoencoderKLOutput:
        """Encode the input samples."""
        if self.config.patch_size:
            args = {"p": self.config.patch_size, "q": self.config.patch_size}
            x = rearrange(x, "b c t (h p) (w q) -> b (c q p) t h w", **args)
        for i in range(1 + (x.shape[2] - 1) // 4):
            args = (self.enc_feat_map, [0])
            if i == 0:
                z = self.encoder(x[:, :, :1], *args)
            else:
                z = torch.cat([z, self.encoder(x[:, :, 4 * i - 3 : 4 * i + 1], *args)], dim=2)
        z, _ = self.quant_conv(z), self.clear_cache()
        posterior = self.latent_dist(z)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z) -> DecoderOutput:
        z = self.post_quant_conv(z)
        for i in range(z.shape[2]):
            args = (self.feat_map, [0], i == 0)
            if i == 0:
                x = self.decoder(z[:, :, i : i + 1], *args)
            else:
                x = torch.cat([x, self.decoder(z[:, :, i : i + 1], *args)], dim=2)
        if self.config.patch_size:
            args = {"p": self.config.patch_size, "q": self.config.patch_size}
            x = rearrange(x, "b (c q p) t h w -> b c t (h p) (w q)", **args)
        self.clear_cache()
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
