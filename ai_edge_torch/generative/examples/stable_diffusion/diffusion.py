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

import torch
from torch import nn
from torch.nn import functional as F

from ai_edge_torch.generative.examples.stable_diffusion.attention import CrossAttention  # NOQA
from ai_edge_torch.generative.examples.stable_diffusion.attention import SelfAttention  # NOQA


class TimeEmbedding(nn.Module):

  def __init__(self, n_embd):
    super().__init__()
    self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
    self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

  def forward(self, x):
    x = self.linear_1(x)
    x = F.silu(x)
    x = self.linear_2(x)
    return x


class ResidualBlock(nn.Module):

  def __init__(self, in_channels, out_channels, n_time=1280):
    super().__init__()
    self.groupnorm_feature = nn.GroupNorm(32, in_channels)
    self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.linear_time = nn.Linear(n_time, out_channels)

    self.groupnorm_merged = nn.GroupNorm(32, out_channels)
    self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(
          in_channels, out_channels, kernel_size=1, padding=0
      )

  def forward(self, feature, time):
    residue = feature

    feature = self.groupnorm_feature(feature)
    feature = F.silu(feature)
    feature = self.conv_feature(feature)

    time = F.silu(time)
    time = self.linear_time(time)

    merged = feature + time.unsqueeze(-1).unsqueeze(-1)
    merged = self.groupnorm_merged(merged)
    merged = F.silu(merged)
    merged = self.conv_merged(merged)

    return merged + self.residual_layer(residue)


class AttentionBlock(nn.Module):

  def __init__(self, n_head: int, n_embd: int, d_context=768):
    super().__init__()
    channels = n_head * n_embd

    self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
    self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    self.layernorm_1 = nn.LayerNorm(channels)
    self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
    self.layernorm_2 = nn.LayerNorm(channels)
    self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
    self.layernorm_3 = nn.LayerNorm(channels)
    self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
    self.linear_geglu_2 = nn.Linear(4 * channels, channels)

    self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

  def forward(self, x, context):
    residue_long = x

    x = self.groupnorm(x)
    x = self.conv_input(x)

    n, c, h, w = x.shape
    x = x.view((n, c, h * w))  # (n, c, hw)
    x = x.transpose(-1, -2)  # (n, hw, c)

    residue_short = x
    x = self.layernorm_1(x)
    x = self.attention_1(x)
    x += residue_short

    residue_short = x
    x = self.layernorm_2(x)
    x = self.attention_2(x, context)
    x += residue_short

    residue_short = x
    x = self.layernorm_3(x)
    x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
    x = x * F.gelu(gate)
    x = self.linear_geglu_2(x)
    x += residue_short

    x = x.transpose(-1, -2)  # (n, c, hw)
    x = x.view((n, c, h, w))  # (n, c, h, w)

    return self.conv_output(x) + residue_long


class Upsample(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

  def forward(self, x):
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    return self.conv(x)


class SwitchSequential(nn.Sequential):

  def forward(self, x, context, time):
    for layer in self:
      if isinstance(layer, AttentionBlock):
        x = layer(x, context)
      elif isinstance(layer, ResidualBlock):
        x = layer(x, time)
      else:
        x = layer(x)
    return x


class UNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.encoders = nn.ModuleList(
        [
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ]
    )
    self.bottleneck = SwitchSequential(
        ResidualBlock(1280, 1280),
        AttentionBlock(8, 160),
        ResidualBlock(1280, 1280),
    )

    self.decoders = nn.ModuleList(
        [
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(
                ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)
            ),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(
                ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)
            ),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ]
    )

  def forward(self, x, context, time):
    skip_connections = []
    for layers in self.encoders:
      x = layers(x, context, time)
      skip_connections.append(x)

    x = self.bottleneck(x, context, time)

    # print('x shape:')
    # print(list(x.shape))
    # print('time shape:')
    # print(list(time.shape))

    for layers in self.decoders:
      x = torch.cat((x, skip_connections.pop()), dim=1)
      x = layers(x, context, time)

    return x


# The encoder component.
class UNetEncoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.time_embedding = TimeEmbedding(320)
    self.encoders = nn.ModuleList(
        [
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ]
    )

  def forward(self, x, context, time):
    time_embedding = self.time_embedding(time)
    skip_connections = []
    for layers in self.encoders:
      x = layers(x, context, time_embedding)
      skip_connections.append(x)

    return x, skip_connections, time_embedding


class UNetBottleNeck(nn.Module):

  def __init__(self):
    super().__init__()
    self.bottleneck = SwitchSequential(
        ResidualBlock(1280, 1280),
        AttentionBlock(8, 160),
        ResidualBlock(1280, 1280),
    )

  def forward(self, x, context, time):
    x = self.bottleneck(x, context, time)
    # print('shape')
    # print(list(x.shape))
    return x


# Unet decoder.
class UNetDecoder1(nn.Module):

  def __init__(self):
    super().__init__()
    self.decoders = nn.ModuleList(
        [
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
        ]
    )

  def forward(self, x, context, time, s9, s10, s11, s12):
    x = torch.cat((x, s12), dim=1)
    x = self.decoders[0](x, context, time)
    x = torch.cat((x, s11), dim=1)
    x = self.decoders[1](x, context, time)
    x = torch.cat((x, s10), dim=1)
    x = self.decoders[2](x, context, time)
    x = torch.cat((x, s9), dim=1)
    x = self.decoders[3](x, context, time)

    return x


class UNetDecoder2(nn.Module):

  def __init__(self):
    super().__init__()
    self.decoders = nn.ModuleList(
        [
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(
                ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)
            ),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
        ]
    )

  def forward(self, x, context, time, s5, s6, s7, s8):
    x = torch.cat((x, s8), dim=1)
    x = self.decoders[0](x, context, time)
    x = torch.cat((x, s7), dim=1)
    x = self.decoders[1](x, context, time)
    x = torch.cat((x, s6), dim=1)
    x = self.decoders[2](x, context, time)
    x = torch.cat((x, s5), dim=1)
    x = self.decoders[3](x, context, time)
    return x


class UNetDecoder3(nn.Module):

  def __init__(self):
    super().__init__()
    self.decoders = nn.ModuleList(
        [
            SwitchSequential(
                ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)
            ),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ]
    )
    self.final = FinalLayer(320, 4)

  def forward(self, x, context, time, s1, s2, s3, s4):
    x = torch.cat((x, s4), dim=1)
    x = self.decoders[0](x, context, time)
    x = torch.cat((x, s3), dim=1)
    x = self.decoders[1](x, context, time)
    x = torch.cat((x, s2), dim=1)
    x = self.decoders[2](x, context, time)
    x = torch.cat((x, s1), dim=1)
    x = self.decoders[3](x, context, time)

    x = self.final(x)
    return x


class UNetDecoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.decoders = nn.ModuleList(
        [
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(
                ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)
            ),
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(
                ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)
            ),
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ]
    )
    self.final = FinalLayer(320, 4)

  def forward(
      self, x, context, time, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12
  ):
    x = torch.cat((x, s12), dim=1)
    x = self.decoders[0](x, context, time)
    x = torch.cat((x, s11), dim=1)
    x = self.decoders[1](x, context, time)
    x = torch.cat((x, s10), dim=1)
    x = self.decoders[2](x, context, time)
    x = torch.cat((x, s9), dim=1)
    x = self.decoders[3](x, context, time)
    x = torch.cat((x, s8), dim=1)
    x = self.decoders[4](x, context, time)
    x = torch.cat((x, s7), dim=1)
    x = self.decoders[5](x, context, time)
    x = torch.cat((x, s6), dim=1)
    x = self.decoders[6](x, context, time)
    x = torch.cat((x, s5), dim=1)
    x = self.decoders[7](x, context, time)
    x = torch.cat((x, s4), dim=1)
    x = self.decoders[0](x, context, time)
    x = torch.cat((x, s3), dim=1)
    x = self.decoders[1](x, context, time)
    x = torch.cat((x, s2), dim=1)
    x = self.decoders[2](x, context, time)
    x = torch.cat((x, s1), dim=1)
    x = self.decoders[3](x, context, time)

    x = self.final(x)

    return x


class FinalLayer(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.groupnorm = nn.GroupNorm(32, in_channels)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.groupnorm(x)
    x = F.silu(x)
    x = self.conv(x)
    return x


class Diffusion(nn.Module):

  def __init__(self):
    super().__init__()
    self.time_embedding = TimeEmbedding(320)
    self.unet = UNet()
    self.final = FinalLayer(320, 4)

  @torch.inference_mode
  def forward(self, latent, context, time):
    time = self.time_embedding(time)
    # print('time:')
    # print(list(time.shape))
    output = self.unet(latent, context, time)
    output = self.final(output)
    return output


# Calling code as if Diffusion is splitted into two parts.
class DiffusionSplitted(nn.Module):

  def __init__(self):
    super().__init__()
    self.unet_encoder = UNetEncoder()
    self.bottleneck = UNetBottleNeck()
    self.unet_decoder1 = UNetDecoder1()
    self.unet_decoder2 = UNetDecoder2()
    self.unet_decoder3 = UNetDecoder3()

  def get_skip_connections(self, latent, context, time):
    _, skip_connections, _ = self.unet_encoder(latent, context, time)
    return skip_connections

  def forward(self, latent, context, time):
    output, skip_connections, time = self.unet_encoder(latent, context, time)
    # print("output shape of unet encoder...")
    # print(list(output.shape))
    # print("output shape of time...")
    # print(list(time.shape))
    output = self.bottleneck(output, context, time)
    # print("output shape of bn")
    # print(list(output.shape))
    output = self.unet_decoder1(
        output,
        context,
        time,
        skip_connections[8],
        skip_connections[9],
        skip_connections[10],
        skip_connections[11],
    )
    # print("output shape of d1:")
    # print(list(output.shape))

    output = self.unet_decoder2(
        output,
        context,
        time,
        skip_connections[4],
        skip_connections[5],
        skip_connections[6],
        skip_connections[7],
    )

    # print("output shape of d2:")
    # print(list(output.shape))
    output = self.unet_decoder3(
        output,
        context,
        time,
        skip_connections[0],
        skip_connections[1],
        skip_connections[2],
        skip_connections[3],
    )
    return output
