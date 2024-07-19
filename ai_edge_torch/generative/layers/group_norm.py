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
  x = torch.permute(x, (0, 2, 3, 1))

  builder = StableHLOCompositeBuilder(
      name="odml.group_norm", attr={"num_groups": num_groups, "eps": eps}
  )
  x = builder.mark_inputs(x)
  x = torch.permute(x, (0, 3, 1, 2))
  y = F.group_norm(x, num_groups, eps=eps)
  y = torch.permute(y, (0, 2, 3, 1))
  y = builder.mark_outputs(y)

  y = torch.permute(y, (0, 3, 1, 2))
  return y
