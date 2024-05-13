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
# `nn.Module` which implements a KV cache.

import torch
from torch import nn
import torch_xla

from ai_edge_torch.hlfb import StableHLOCompositeBuilder


class KVCache(nn.Module):

  def __init__(self, batch_size, kv_cache_max, n_heads, head_dim, enable_hlfb=False):
    """Initializes the KVCache layer.

    Args:
      batch_size (int): batch size. Currently only batch size 1 is supported.
      kv_cache_max (int): the max length of KV cache.
      n_heads (int): number of kv heads.
      head_dim (int): the head dimension size.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__()
    cache_shape = (batch_size, kv_cache_max, n_heads, head_dim)
    self.register_buffer("k_cache", torch.zeros(cache_shape), persistent=False)
    self.register_buffer("v_cache", torch.zeros(cache_shape), persistent=False)
    self.enable_hlfb = enable_hlfb
    self.kv_cache_max = kv_cache_max

  def update_cache(self, input_pos, k_val, v_val):
    """Update an entry in the KV cache.

    Args:
      input_pos (torch.Tensor): the input position.
      k_val (torch.Tensor): the new `key` value.
      v_val (torch.Tensor): the new `value` value.

    Returns:
      The updated key and value tensor.
    """
    if self.enable_hlfb:
      return self.update_cache_with_hlfb(input_pos, k_val, v_val)

    updated_k = self.k_cache.index_copy_(1, input_pos, k_val)
    updated_v = self.v_cache.index_copy_(1, input_pos, v_val)
    # Here we need a clone otherwise dynamo export will fail.
    return torch.clone(updated_k), torch.clone(updated_v)

  def update_cache_with_hlfb(self, input_pos, k_val, v_val):
    """Update an entry in the KV cache and enable high-level function boundary.

    Args:
      input_pos (torch.Tensor): the input position.
      k_val (torch.Tensor): the new `key` value.
      v_val (torch.Tensor): the new `value` value.

    Returns:
      The updated key and value tensor.
    """

    builder = StableHLOCompositeBuilder(
        name="odml.update_kv_cache", attr={"kv_cache_max": self.kv_cache_max}
    )
    k_cache, v_cache, input_pos, k_val, v_val = builder.mark_inputs(
        self.k_cache, self.v_cache, input_pos, k_val, v_val
    )
    updated_k = k_cache.index_copy_(1, input_pos, k_val)
    updated_v = v_cache.index_copy_(1, input_pos, v_val)
    updated_k, updated_v = builder.mark_outputs(updated_k, updated_v)
    return updated_k, updated_v
