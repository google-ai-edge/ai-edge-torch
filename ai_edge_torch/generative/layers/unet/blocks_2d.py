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

from typing import Optional

import torch
from torch import nn

from ai_edge_torch.generative.layers.attention import SelfAttention
import ai_edge_torch.generative.layers.builder as layers_builder
import ai_edge_torch.generative.layers.unet.builder as unet_builder
import ai_edge_torch.generative.layers.unet.model_config as unet_cfg


class ResidualBlock2D(nn.Module):
  """2D Residual block containing two Conv2D with optional time embedding as input."""

  def __init__(self, config: unet_cfg.ResidualBlock2DConfig):
    """Initialize an instance of the ResidualBlock2D.

    Args:
      config (unet_cfg.ResidualBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    self.norm_1 = layers_builder.build_norm(
        config.in_channels, config.normalization_config
    )
    self.conv_1 = nn.Conv2d(
        config.in_channels, config.out_channels, kernel_size=3, stride=1, padding=1
    )
    if config.time_embedding_channels is not None:
      self.time_emb_proj = nn.Linear(
          config.time_embedding_channels, config.out_channels
      )
    else:
      self.time_emb_proj = None
    self.norm_2 = layers_builder.build_norm(
        config.out_channels, config.normalization_config
    )
    self.conv_2 = nn.Conv2d(
        config.out_channels, config.out_channels, kernel_size=3, stride=1, padding=1
    )
    self.act_fn = layers_builder.get_activation(config.activation_config)
    if config.in_channels == config.out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(
          config.in_channels, config.out_channels, kernel_size=1, stride=1, padding=0
      )

  def forward(
      self, input_tensor: torch.Tensor, time_emb: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """Forward function of the ResidualBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (Optional[torch.Tensor]): optional time embedding tensor.

    Returns:
      output hidden_states tensor after ResidualBlock2D.
    """
    residual = input_tensor
    x = self.norm_1(input_tensor)
    x = self.act_fn(x)
    x = self.conv_1(x)
    if self.time_emb_proj is not None:
      time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
      x = x + time_emb
    x = self.norm_2(x)
    x = self.act_fn(x)
    x = self.conv_2(x)
    x = x + self.residual_layer(residual)
    return x


class AttentionBlock2D(nn.Module):
  """2D self attention block

  x = SelfAttention(Norm(input_tensor))

  """

  def __init__(self, config: unet_cfg.AttentionBlock2DConfig):
    """Initialize an instance of the AttentionBlock2D.

    Args:
      config (unet_cfg.AttentionBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.norm = layers_builder.build_norm(config.dims, config.normalization_config)
    self.attention = SelfAttention(config.dims, config.attention_config, 0, True)

  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """Forward function of the AttentionBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.

    Returns:
      output activation tensor after self attention.
    """
    residual = input_tensor
    x = self.norm(input_tensor)
    B, C, H, W = x.shape
    x = x.view(B, C, H * W)
    x = x.transpose(-1, -2)
    x = self.attention(x)
    x = x.transpose(-1, -2)
    x = x.view(B, C, H, W)
    x = x + residual
    return x


class UpDecoderBlock2D(nn.Module):
  """Decoder block containing several residual blocks followed by an optional upsampler.

       input_tensor
            |
            ▼
  ┌───────────────────┐
  │  ResidualBlock2D  │ num_layers
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │    (Optional)     │
  │     Upsampler     │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │    (Optional)     │
  │      Conv2D       │
  └─────────┬─────────┘
            │
            ▼
      hidden_states
  """

  def __init__(self, config: unet_cfg.UpDecoderBlock2DConfig):
    """Initialize an instance of the UpDecoderBlock2D.

    Args:
      config (unet_cfg.UpDecoderBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    resnets = []
    for i in range(config.num_layers):
      input_channels = config.in_channels if i == 0 else config.out_channels
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=input_channels,
                  out_channels=config.out_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
    self.resnets = nn.ModuleList(resnets)
    if config.add_upsample:
      self.upsampler = unet_builder.build_upsampling(config.sampling_config)
      if config.upsample_conv:
        self.upsample_conv = nn.Conv2d(
            config.out_channels, config.out_channels, kernel_size=3, stride=1, padding=1
        )
    else:
      self.upsampler = None

  def forward(
      self, input_tensor: torch.Tensor, time_emb: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """Forward function of the UpDecoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is configured to accept
        time embedding context.

    Returns:
      output hidden_states tensor after UpDecoderBlock2D.
    """
    hidden_states = input_tensor
    for resnet in self.resnets:
      hidden_states = resnet(hidden_states, time_emb)
    if self.upsampler:
      hidden_states = self.upsampler(hidden_states)
      if self.upsample_conv:
        hidden_states = self.upsample_conv(hidden_states)
    return hidden_states


class MidBlock2D(nn.Module):
  """Middle block containing at least one residual blocks with optional interleaved attention blocks.

           input_tensor
                |
                ▼
      ┌───────────────────┐
      │  ResidualBlock2D  │
      └─────────┬─────────┘
                │
  ┌─────────────▼─────────────┐
  │   ┌───────────────────┐   │
  │   │    (Optional)     │   │
  │   │  AttentionBlock2D │   │
  │   └─────────┬─────────┘   │  num_layers
  │             │             │
  │   ┌─────────▼─────────┐   │
  │   │  ResidualBlock2D  │   │
  │   └───────────────────┘   │
  └─────────────┬─────────────┘
                │
                ▼
          hidden_states
  """

  def __init__(self, config: unet_cfg.MidBlock2DConfig):
    """Initialize an instance of the MidBlock2D.

    Args:
      config (unet_cfg.MidBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    resnets = [
        ResidualBlock2D(
            unet_cfg.ResidualBlock2DConfig(
                in_channels=config.in_channels,
                out_channels=config.in_channels,
                time_embedding_channels=config.time_embedding_channels,
                normalization_config=config.normalization_config,
                activation_config=config.activation_config,
            )
        )
    ]
    attentions = []
    for i in range(config.num_layers):
      if self.config.attention_block_config:
        attentions.append(AttentionBlock2D(config.attention_block_config))
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=config.in_channels,
                  out_channels=config.in_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
    self.resnets = nn.ModuleList(resnets)
    self.attentions = nn.ModuleList(attentions)

  def forward(
      self, input_tensor: torch.Tensor, time_emb: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    """Forward function of the MidBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is configured to accept
        time embedding context.

    Returns:
      output hidden_states tensor after MidBlock2D.
    """
    hidden_states = self.resnets[0](input_tensor, time_emb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      if attn is not None:
        hidden_states = attn(hidden_states)
      hidden_states = resnet(hidden_states, time_emb)
    return hidden_states
