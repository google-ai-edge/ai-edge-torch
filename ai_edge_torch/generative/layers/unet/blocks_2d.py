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

from typing import List, Optional

import torch
from torch import nn

from ai_edge_torch.generative.layers.attention import CrossAttention
from ai_edge_torch.generative.layers.attention import SelfAttention
import ai_edge_torch.generative.layers.builder as layers_builder
import ai_edge_torch.generative.layers.model_config as layers_cfg
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
      time_emb = self.act_fn(time_emb)
      time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
      x = x + time_emb
    x = self.norm_2(x)
    x = self.act_fn(x)
    x = self.conv_2(x)
    x = x + self.residual_layer(residual)
    return x


class AttentionBlock2D(nn.Module):
  """2D self attention block

  residual = x
  x = SelfAttention(Norm(input_tensor)) + residual

  """

  def __init__(self, config: unet_cfg.AttentionBlock2DConfig):
    """Initialize an instance of the AttentionBlock2D.

    Args:
      config (unet_cfg.AttentionBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.norm = layers_builder.build_norm(config.dim, config.normalization_config)
    self.attention = SelfAttention(config.dim, config.attention_config, 0, True)

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


class CrossAttentionBlock2D(nn.Module):
  """2D cross attention block

  residual = x
  x = CrossAttention(Norm(input_tensor), context) + residual

  """

  def __init__(self, config: unet_cfg.CrossAttentionBlock2DConfig):
    """Initialize an instance of the AttentionBlock2D.

    Args:
      config (unet_cfg.CrossAttentionBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    self.norm = layers_builder.build_norm(config.query_dim, config.normalization_config)
    self.attention = CrossAttention(
        config.query_dim, config.cross_dim, config.attention_config, 0, True
    )

  def forward(
      self, input_tensor: torch.Tensor, context_tensor: torch.Tensor
  ) -> torch.Tensor:
    """Forward function of the CrossAttentionBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      context_tensor (torch.Tensor): the context tensor to apply cross attention on.

    Returns:
      output activation tensor after cross attention.
    """
    residual = input_tensor
    x = self.norm(input_tensor)
    B, C, H, W = x.shape
    x = x.view(B, C, H * W)
    x = x.transpose(-1, -2)
    x = self.attention(x, context_tensor)
    x = x.transpose(-1, -2)
    x = x.view(B, C, H, W)
    x = x + residual
    return x


class FeedForwardBlock2D(nn.Module):
  """2D feed forward block

  residual = x
  x = w2(Activation(w1(Norm(x)))) + residual

  """

  def __init__(
      self,
      config: unet_cfg.FeedForwardBlock2DConfig,
  ):
    super().__init__()
    self.config = config
    self.act = layers_builder.get_activation(config.activation_config)
    self.norm = layers_builder.build_norm(config.dim, config.normalization_config)
    if config.activation_config.type == layers_cfg.ActivationType.GE_GLU:
      self.w1 = nn.Identity()
      self.w2 = nn.Linear(config.hidden_dim, config.dim)
    else:
      self.w1 = nn.Linear(config.dim, config.hidden_dim)
      self.w2 = nn.Linear(config.hidden_dim, config.dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x

    B, C, H, W = x.shape
    x = x.view((B, C, H * W))
    x = x.transpose(-1, -2)  # (B, HW, C)

    x = self.norm(x)
    x = self.w1(x)
    x = self.act(x)
    x = self.w2(x)

    x = x.transpose(-1, -2)  # (B, C, HW)
    x = x.view((B, C, H, W))

    return x + residual


class TransformerBlock2D(nn.Module):
  """Basic transformer block used in UNet of diffusion model

       input_tensor    context_tensor
            |                 |
  ┌─────────▼─────────┐       |
  │      ConvIn       |       │
  └─────────┬─────────┘       |
            |                 |
            ▼                 |
  ┌───────────────────┐       |
  │  Attention Block  │       |
  └─────────┬─────────┘       |
            │                 |
  ┌────────────────────┐      |
  │CrossAttention Block│◄─────┘
  └─────────┬──────────┘
            │
  ┌─────────▼─────────┐
  │ FeedForwardBlock  │
  └─────────┬─────────┘
            │
  ┌─────────▼─────────┐
  │      ConvOut      │
  └─────────┬─────────┘
            ▼
      hidden_states


  """

  def __init__(self, config: unet_cfg.TransformerBlock2Dconfig):
    """Initialize an instance of the TransformerBlock2D.

    Args:
      config (unet_cfg.TransformerBlock2Dconfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    self.pre_conv_norm = layers_builder.build_norm(
        config.attention_block_config.dim, config.pre_conv_normalization_config
    )
    self.conv_in = nn.Conv2d(
        config.attention_block_config.dim,
        config.attention_block_config.dim,
        kernel_size=1,
        padding=0,
    )
    self.self_attention = AttentionBlock2D(config.attention_block_config)
    self.cross_attention = CrossAttentionBlock2D(config.cross_attention_block_config)
    self.feed_forward = FeedForwardBlock2D(config.feed_forward_block_config)
    self.conv_out = nn.Conv2d(
        config.attention_block_config.dim,
        config.attention_block_config.dim,
        kernel_size=1,
        padding=0,
    )

  def forward(self, x: torch.Tensor, context: torch.Tensor):
    """Forward function of the TransformerBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      context_tensor (torch.Tensor): the context tensor to apply cross attention on.

    Returns:
      output activation tensor after transformer block.
    """
    residual_long = x

    x = self.pre_conv_norm(x)
    x = self.conv_in(x)

    x = self.self_attention(x)
    x = self.cross_attention(x, context)
    x = self.feed_forward(x)

    x = self.conv_out(x)
    x = x + residual_long

    return x


class DownEncoderBlock2D(nn.Module):
  """Encoder block containing several residual blocks with optional interleaved transformer blocks.

            input_tensor
                 |
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │  num_layers
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  └──────────────┬─────────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │     Downsampler    │
      └──────────┬─────────┘
                 │
                 ▼
           hidden_states
  """

  def __init__(self, config: unet_cfg.DownEncoderBlock2DConfig):
    """Initialize an instance of the DownEncoderBlock2D.

    Args:
      config (unet_cfg.DownEncoderBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    resnets = []
    transformers = []
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
      if config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
    self.resnets = nn.ModuleList(resnets)
    self.transformers = nn.ModuleList(transformers)
    if config.add_downsample:
      self.downsampler = unet_builder.build_downsampling(config.sampling_config)
    else:
      self.downsampler = None

  def forward(
      self,
      input_tensor: torch.Tensor,
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the DownEncoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is configured to accept
        time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if configured to use transofrmer block.

    Returns:
      output hidden_states tensor after DownEncoderBlock2D.
    """
    hidden_states = input_tensor
    for resnet, transformer in zip(self.resnets, self.transformers):
      hidden_states = resnet(hidden_states, time_emb)
      if transformer is not None:
        hidden_states = transformer(hidden_states, context_tensor)
    if self.downsampler:
      hidden_states = self.downsampler(hidden_states)
    return hidden_states


class UpDecoderBlock2D(nn.Module):
  """Decoder block containing several residual blocks with optional interleaved transformer blocks.

            input_tensor
                 |
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │  num_layers
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  └──────────────┬─────────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │      Upsampler     │
      └──────────┬─────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │       Conv2D       │
      └──────────┬─────────┘
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
    transformers = []
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
      if config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
    self.resnets = nn.ModuleList(resnets)
    self.transformers = nn.ModuleList(transformers)
    if config.add_upsample:
      self.upsampler = unet_builder.build_upsampling(config.sampling_config)
      if config.upsample_conv:
        self.upsample_conv = nn.Conv2d(
            config.out_channels, config.out_channels, kernel_size=3, stride=1, padding=1
        )
    else:
      self.upsampler = None

  def forward(
      self,
      input_tensor: torch.Tensor,
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the UpDecoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is configured to accept
        time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if configured to use transofrmer block.

    Returns:
      output hidden_states tensor after UpDecoderBlock2D.
    """
    hidden_states = input_tensor
    for resnet, transformer in zip(self.resnets, self.transformers):
      hidden_states = resnet(hidden_states, time_emb)
      if transformer is not None:
        hidden_states = transformer(hidden_states, context_tensor)
    if self.upsampler:
      hidden_states = self.upsampler(hidden_states)
      if self.upsample_conv:
        hidden_states = self.upsample_conv(hidden_states)
    return hidden_states


class SkipUpDecoderBlock2D(nn.Module):
  """Decoder block contains skip connections and residual blocks with optional interleaved transformer blocks.

   input_tensor, skip_connection_tensors
                 |
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │  num_layers
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  └──────────────┬─────────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │      Upsampler     │
      └──────────┬─────────┘
                 │
      ┌──────────▼─────────┐
      │     (Optional)     │
      │       Conv2D       │
      └──────────┬─────────┘
                 │
                 ▼
           hidden_states
  """

  def __init__(self, config: unet_cfg.SkipUpDecoderBlock2DConfig):
    """Initialize an instance of the SkipUpDecoderBlock2D.

    Args:
      config (unet_cfg.SkipUpDecoderBlock2DConfig): the configuration of this block.
    """
    super().__init__()
    self.config = config
    resnets = []
    transformers = []
    for i in range(config.num_layers):
      res_skip_channels = (
          config.in_channels if (i == config.num_layers - 1) else config.out_channels
      )
      resnet_in_channels = config.prev_out_channels if i == 0 else config.out_channels
      resnets.append(
          ResidualBlock2D(
              unet_cfg.ResidualBlock2DConfig(
                  in_channels=resnet_in_channels + res_skip_channels,
                  out_channels=config.out_channels,
                  time_embedding_channels=config.time_embedding_channels,
                  normalization_config=config.normalization_config,
                  activation_config=config.activation_config,
              )
          )
      )
      if config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
    self.resnets = nn.ModuleList(resnets)
    self.transformers = nn.ModuleList(transformers)
    if config.add_upsample:
      self.upsampler = unet_builder.build_upsampling(config.sampling_config)
      if config.upsample_conv:
        self.upsample_conv = nn.Conv2d(
            config.out_channels, config.out_channels, kernel_size=3, stride=1, padding=1
        )
    else:
      self.upsampler = None

  def forward(
      self,
      input_tensor: torch.Tensor,
      skip_connection_tensors: List[torch.Tensor],
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the SkipUpDecoderBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      skip_connection_tensors (List[torch.Tensor]): the skip connection tensors from encoder blocks.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is configured to accept
        time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if configured to use transofrmer block.

    Returns:
      output hidden_states tensor after SkipUpDecoderBlock2D.
    """
    hidden_states = input_tensor
    for resnet, skip_connection_tensor, transformer in zip(
        self.resnets, skip_connection_tensors, self.transformers
    ):
      hidden_states = torch.cat([resnet, skip_connection_tensor], dim=1)
      hidden_states = resnet(hidden_states, time_emb)
      if transformer is not None:
        hidden_states = transformer(hidden_states, context_tensor)
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
  ┌──────────────▼─────────────┐
  │   ┌────────────────────┐   │
  │   │     (Optional)     │   │
  │   │  AttentionBlock2D  │   │
  │   └──────────┬─────────┘   │
  │              │             │
  │   ┌──────────▼─────────┐   │
  │   │     (Optional)     │   │  num_layers
  │   │ TransformerBlock2D │   │
  │   └──────────┬─────────┘   │
  │              │             │
  │   ┌──────────▼─────────┐   │
  │   │   ResidualBlock2D  │   │
  │   └────────────────────┘   │
  └──────────────┬─────────────┘
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
    transformers = []
    for i in range(config.num_layers):
      if self.config.attention_block_config:
        attentions.append(AttentionBlock2D(config.attention_block_config))
      if self.config.transformer_block_config:
        transformers.append(TransformerBlock2D(config.transformer_block_config))
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
    self.transformers = nn.ModuleList(transformers)

  def forward(
      self,
      input_tensor: torch.Tensor,
      time_emb: Optional[torch.Tensor] = None,
      context_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the MidBlock2D.

    Args:
      input_tensor (torch.Tensor): the input tensor.
      time_emb (torch.Tensor): optional time embedding tensor, if the block is configured to accept
        time embedding.
      context_tensor (torch.Tensor): optional context tensor, if the block if configured to use
        transofrmer block.

    Returns:
      output hidden_states tensor after MidBlock2D.
    """
    hidden_states = self.resnets[0](input_tensor, time_emb)
    for attn, transformer, resnet in zip(
        self.attentions, self.transformers, self.resnets[1:]
    ):
      if attn is not None:
        hidden_states = attn(hidden_states)
      if transformer is not None:
        hidden_states = transformer(hidden_states, context_tensor)
      hidden_states = resnet(hidden_states, time_emb)
    return hidden_states
