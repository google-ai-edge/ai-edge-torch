# Copyright 2024 The AI Edge Torch Authors.
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
# ==============================================================================

# UNet configuration class.
from dataclasses import dataclass
from dataclasses import field
import enum
from typing import List, Optional

import ai_edge_torch.generative.layers.model_config as layers_cfg


@dataclass
class SamplingType(enum.Enum):
  NEAREST = enum.auto()
  BILINEAR = enum.auto()


@dataclass
class SamplingConfig:
  scale_factor: float
  mode: SamplingType


@dataclass
class ResidualBlock2DConfig:
  in_channels: int
  out_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  # Optional time embedding channels if the residual block takes a time embedding context as input
  time_embedding_channels: Optional[int] = None


@dataclass
class AttentionBlock2DConfig:
  dims: int
  normalization_config: layers_cfg.NormalizationConfig
  attention_config: layers_cfg.AttentionConfig


@dataclass
class UpDecoderBlock2DConfig:
  in_channels: int
  out_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  num_layers: int
  # Optional time embedding channels if the residual blocks take a time embedding context as input
  time_embedding_channels: Optional[int] = None
  # Whether to add upsample operation after residual blocks
  add_upsample: bool = True
  # Whether to add a conv2d layer after upsample
  upsample_conv: bool = True
  # Optional sampling config if add_upsample is True.
  sampling_config: Optional[SamplingConfig] = None


@dataclass
class MidBlock2DConfig:
  in_channels: int
  normalization_config: layers_cfg.NormalizationConfig
  activation_config: layers_cfg.ActivationConfig
  num_layers: int
  # Optional time embedding channels if the residual blocks take a time embedding context as input
  time_embedding_channels: Optional[int] = None
  # Optional config of attention blocks interleaved with residual blocks
  attention_block_config: Optional[AttentionBlock2DConfig] = None


@dataclass
class AutoEncoderConfig:
  """Configurations of encoder/decoder in the autoencoder model."""

  # The activation type of encoder/decoder blocks.
  activation_config: layers_cfg.ActivationConfig

  # The output channels of each block.
  block_out_channels: List[int]

  # Number of channels in the input image.
  in_channels: int

  # Number of channels in the output.
  out_channels: int

  # Number of channels in the latent space.
  latent_channels: int

  # The component-wise standard deviation of the trained latent space computed using the first batch of the
  # training set. This is used to scale the latent space to have unit variance when training the diffusion
  # model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
  # diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
  # / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
  # Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
  scaling_factor: float

  # The layesr number of each encoder/decoder block.
  layers_per_block: int

  # The normalization config.
  normalization_config: layers_cfg.NormalizationConfig

  # The configuration of middle blocks, that is, after the last block of encoder and before the first block of decoder.
  mid_block_config: MidBlock2DConfig
