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
"""Embedding layers."""

import sys

import torch
from torch import nn


class FlexRotaryEmbedding(nn.Identity):
    """Flexible rotary position embedding layer."""

    class PEFunc(object):
        """Apply RoPE weight to Q/K tensor."""

        def __init__(self, weight: torch.Tensor):
            self.weight = weight

        @torch.compile(fullgraph=True, disable=sys.platform != "linux")
        def interleaved_impl(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return w[..., 0].mul(x[..., 0]).add_(w[..., 1] * x[..., 1]).flatten(3)

        @torch.compile(fullgraph=True, disable=sys.platform != "linux")
        def partitioned_impl(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return w[..., 0].mul(x[:, :, :, 0]).add_(w[..., 1] * x[:, :, :, 1]).flatten(3)

        def __call__(self, x: torch.Tensor, interleaved=False) -> torch.Tensor:
            w = self.weight = self.weight.to(dtype=x.dtype)
            x = x.unflatten(-1, (-1, 1, 2) if interleaved else (2, -1, 1))
            return (self.interleaved_impl if interleaved else self.partitioned_impl)(x, w)

    @staticmethod
    def from_config(config):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return FlexRotaryEmbedding(head_dim, base=config.rope_theta)

    def __init__(self, dim=128, base=10000.0):
        super(FlexRotaryEmbedding, self).__init__()
        self.dim, self.base = dim, base
        self.rep1, self.rep2 = dim // 8, (dim // 2 - dim // 8 * 3) // 2
        self.register_buffer("scale", torch.arange(0, dim, 2).float() / dim, persistent=False)

    def get_pos(self, input_shape, shift=0, has_bov=True) -> torch.Tensor:
        num_blocks = 1 if len(input_shape) < 4 else input_shape[-4]
        block_size = 1 if len(input_shape) < 3 else input_shape[-3]
        grid_shape = [num_blocks * block_size] + list(input_shape[-2:])
        pos = torch.zeros(grid_shape + [3], dtype=torch.int32, device=self.scale.device)
        grid = [torch.arange(_, device=pos.device) for _ in grid_shape]
        [pos[..., i].add_(grid[i].view([-1 if i == j else 1 for j in range(3)])) for i in range(3)]
        pos, device = pos.unflatten(0, (-1, block_size)).flatten(1, 3), pos.device
        bov_pos = torch.arange(num_blocks, device=device).view(-1, 1, 1).repeat(1, 1, 3)
        pos[..., 0] += torch.arange(num_blocks, device=device).view(-1, 1).add_(shift + has_bov)
        return torch.cat([bov_pos.mul(block_size + 1).add(shift), pos], 1) if has_bov else pos

    def get_func(self, pos: torch.Tensor, *args, **kwargs) -> PEFunc:
        t = torch.cat([pos.repeat(1, 1, self.rep1), pos[..., 1:].repeat(1, 1, self.rep2)], -1)
        freq = t * torch.pow(self.base, self.scale.float()).reciprocal_().unsqueeze(0)
        freq = torch.stack([freq.cos(), -freq.sin(), freq.sin(), freq.cos()], dim=-1)
        return self.PEFunc(freq.view(freq.shape[:-1] + (2, 2)).unsqueeze(2))
