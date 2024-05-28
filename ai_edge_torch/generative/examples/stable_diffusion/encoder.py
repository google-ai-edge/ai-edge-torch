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

from ai_edge_torch.generative.examples.stable_diffusion.decoder import AttentionBlock  # NOQA
from ai_edge_torch.generative.examples.stable_diffusion.decoder import ResidualBlock  # NOQA
import ai_edge_torch.generative.utilities.loader as loading_utils


class Encoder(nn.Sequential):

  def __init__(self):
    super().__init__(
        nn.Conv2d(3, 128, kernel_size=3, padding=1),
        ResidualBlock(128, 128),
        ResidualBlock(128, 128),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
        ResidualBlock(128, 256),
        ResidualBlock(256, 256),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
        ResidualBlock(256, 512),
        ResidualBlock(512, 512),
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        ResidualBlock(512, 512),
        AttentionBlock(512),
        ResidualBlock(512, 512),
        nn.GroupNorm(32, 512),
        nn.SiLU(),
        nn.Conv2d(512, 8, kernel_size=3, padding=1),
        nn.Conv2d(8, 8, kernel_size=1, padding=0),
    )

  @torch.inference_mode
  def forward(self, x, noise):
    for module in self:
      if getattr(module, 'stride', None) == (
          2,
          2,
      ):  # Padding at downsampling should be asymmetric (see #8)
        x = F.pad(x, (0, 1, 0, 1))
      x = module(x)

    mean, log_variance = torch.chunk(x, 2, dim=1)
    log_variance = torch.clamp(log_variance, -30, 20)
    variance = log_variance.exp()
    stdev = variance.sqrt()
    x = mean + stdev * noise

    x *= 0.18215
    return x
