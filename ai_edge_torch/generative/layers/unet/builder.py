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
# Builder utils for individual components.

from torch import nn
import torch.nn.functional as F

import ai_edge_torch.generative.layers.unet.model_config as unet_config


def build_upsampling(config: unet_config.SamplingConfig):
  if config.mode == unet_config.SamplingType.NEAREST:
    return nn.UpsamplingNearest2d(scale_factor=config.scale_factor)
  elif config.mode == unet_config.SamplingType.BILINEAR:
    return nn.UpsamplingBilinear2d(scale_factor=config.scale_factor)
  else:
    raise ValueError("Unsupported upsampling type.")
