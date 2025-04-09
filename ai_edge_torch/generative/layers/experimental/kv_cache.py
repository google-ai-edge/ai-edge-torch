# Copyright 2025 The AI Edge Torch Authors.
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

"""Utility functions for KV Cache.

This is an experimental implementation and is subject to change at any time.
"""

from ai_edge_torch.generative.custom_ops import dynamic_update_slice as dus_utils
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch


def update(
    cache: kv_utils.KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> kv_utils.KVCacheEntry:
  """Out of place update of Cache buffer.

  Args:
      cache (kv_utils.KVCacheEntry): The original cache buffer.
      input_pos (torch.Tensor): The update slice positions.
      k_slice (torch.Tensor): The K slice to be updated in the new cache.
      v_slice (torch.Tensor): The V slice to be updated in the new cache.

  Returns:
      kv_utils.KVCacheEntry: The updated KVCacheBase entry based on the passed
      inputs.
  """
  assert (
      cache.kv_layout == kv_utils.KV_LAYOUT_TRANSPOSED
  ), "KV entry must have transposed layout."
  update_kv_cache = _update_kv_impl_transposed
  return update_kv_cache(cache, input_pos, k_slice, v_slice)


def _get_slice_indices(
    positions: torch.Tensor, cache_dim: int, ts_idx: int
) -> torch.Tensor:
  """Returns the slice indices."""
  positions = positions.float()[0].reshape(
      1,
  )

  zeros = torch.zeros((1,), dtype=torch.float32)
  indices = []
  for i in range(cache_dim):
    if i == ts_idx:
      indices.append(positions)
    else:
      indices.append(zeros)
  slice_indices = torch.cat(indices, dim=0)
  slice_indices = slice_indices.int()
  return slice_indices


def _update_kv_impl_transposed(
    cache: kv_utils.KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> kv_utils.KVCacheEntry:
  """Update the cache buffer with High Level Function Boundary annotation."""
  cache_dim = 4
  k_ts_idx = 2
  v_ts_idx = 3
  positions = input_pos.clone()
  k_slice_indices = _get_slice_indices(positions, cache_dim, k_ts_idx)
  v_slice_indices = _get_slice_indices(positions, cache_dim, v_ts_idx)
  k = dus_utils.dynamic_update_slice(
      cache.k_cache, k_slice, [x for x in k_slice_indices]
  )
  v = dus_utils.dynamic_update_slice(
      cache.v_cache, v_slice, [x for x in v_slice_indices]
  )
  return kv_utils.KVCacheEntry(k, v, cache.kv_layout)
