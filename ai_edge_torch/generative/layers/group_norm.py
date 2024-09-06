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
    w: torch.Tensor,
    b: torch.Tensor,
    num_groups: int,
    eps: float,
):
  # Let layout_rewrite.py builds NHWC group norm with composite
  y = F.group_norm(x, num_groups, weight=w, bias=b, eps=eps)
  return y

def layer_norm_with_hlfb(
    x: torch.Tensor,
    dim: int,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
):
  builder = StableHLOCompositeBuilder(
      name="odml.layer_norm", attr={"eps": eps}
  )
  x, w, b = builder.mark_inputs(x, w, b)
  y = F.layer_norm(x, x.shape, weight=w.broadcast_to(x.shape), bias=b.broadcast_to(x.shape), eps=eps)
  y = builder.mark_outputs(y)
  return y
