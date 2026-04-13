# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Simple implementation of AutoEncoderKL for Cosmos3D."""

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.modeling_utils import IdentityDistribution
from diffnext.models.autoencoders.modeling_utils import DecoderOutput, TilingMixin
from diffnext.models.autoencoders.autoencoder_vq_cosmos3d import Encoder, Decoder, Conv3d


class AutoencoderKLCosmos3D(ModelMixin, ConfigMixin, TilingMixin):
    """AutoEncoder KL."""

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock3D",) * 3,
        up_block_types=("UpDecoderBlock3D",) * 3,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=1,
        sample_size=512,
        sample_frames=49,
        scaling_factor=1.8628,
        shift_factor=None,
        force_upcast=False,
        patch_size=2,
        temporal_stride=4,
        spatial_stride=8,
    ):
        super(AutoencoderKLCosmos3D, self).__init__()
        latent_min_t = (sample_frames - 1) // temporal_stride + 1
        TilingMixin.__init__(self, sample_frames, latent_min_t=latent_min_t, sample_ovr_t=1)
        extra_args = {"patch_size": patch_size}
        extra_args.update({"temporal_stride": temporal_stride, "spatial_stride": spatial_stride})
        channels, layers = block_out_channels, layers_per_block
        self.encoder = Encoder(in_channels, latent_channels, channels, layers, **extra_args)
        self.decoder = Decoder(latent_channels, out_channels, channels, layers, **extra_args)
        self.quant_conv = Conv3d(latent_channels, latent_channels, 1)
        self.post_quant_conv = Conv3d(latent_channels, latent_channels, 1)
        self.latent_dist = IdentityDistribution

    def scale_(self, x):
        x.add_(-self.config.shift_factor) if self.config.shift_factor else None
        return x.mul_(self.config.scaling_factor)

    def unscale_(self, x):
        x.mul_(1 / self.config.scaling_factor)
        return x.add_(self.config.shift_factor) if self.config.shift_factor else x

    def encode(self, x) -> AutoencoderKLOutput:
        """Encode the input samples."""
        z = self.tiled_encoder(self.forward(x))
        z = self.quant_conv(z)
        posterior = self.latent_dist(z)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z) -> DecoderOutput:
        """Decode the input indices."""
        extra_dim = 2 if z.dim() == 4 else None
        z = z.unsqueeze_(extra_dim) if extra_dim is not None else z
        z = self.post_quant_conv(self.forward(z))
        x = self.tiled_decoder(z)
        x = x.squeeze_(extra_dim) if extra_dim is not None else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
