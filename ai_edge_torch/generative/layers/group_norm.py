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


# class KVCache(nn.Module):
    
#   def __init__(self, num_groups: int,
#     num_channels: int,
#     eps: float):
#     super().__init__()
#     self.num_groups = num_groups
#     self.num_channels = num_channels
#     self.eps = eps

#   def update_cache(self, input_pos, k_val, v_val):
#     """Update an entry in the KV cache.

#     Args:
#       input_pos (torch.Tensor): the input position.
#       k_val (torch.Tensor): the new `key` value.
#       v_val (torch.Tensor): the new `value` value.

#     Returns:
#       The updated key and value tensor.
#     """
#     if self.enable_hlfb:
#       return self.update_cache_with_hlfb(input_pos, k_val, v_val)

#     updated_k = self.k_cache.index_copy_(1, input_pos, k_val)
#     updated_v = self.v_cache.index_copy_(1, input_pos, v_val)
#     # Here we need a clone otherwise dynamo export will fail.
#     return torch.clone(updated_k), torch.clone(updated_v)

def group_norm_with_hlfb(
    x: torch.Tensor,
    num_groups: int,
    eps: float,
):

  builder = StableHLOCompositeBuilder(
      name="odml.group_norm", attr={"num_groups": num_groups, "eps": eps}
  )
  x = builder.mark_inputs(x)

  # x = x.permute(0, 3, 1, 2)
  y = F.group_norm(x, num_groups, eps=eps)
  # x = x.permute(0, 2, 3, 1)

  y = builder.mark_outputs(y)
  return y
