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
# Implements scaled dot product attention.

from typing import Optional

import torch
# from torch import nn
import torch.nn.functional as F

from ai_edge_torch.hlfb import StableHLOCompositeBuilder


def group_norm_with_hlfb(
    x: torch.Tensor,
    num_groups: int,
    eps: float,
):

  B, C, H, W = x.shape
  x = x.view(B, C, H * W)
  x = x.transpose(-1, -2)
  x = x.view(B, H, W, C)

  builder = StableHLOCompositeBuilder(
      name="odml.group_norm", attr={"num_groups": num_groups, "eps": eps}
  )
  x = builder.mark_inputs(x)

  y = F.group_norm(x, num_groups, eps=eps)

  y = builder.mark_outputs(y)

  B, H, H, C = y.shape
  y = y.view(B, H * W, C)
  y = y.transpose(-1, -2)
  y = y.view(B, C, H, W)
  return y
