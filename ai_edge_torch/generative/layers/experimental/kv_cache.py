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
# Cache management utilities.

from dataclasses import dataclass
from typing import Tuple

import torch

from ai_edge_torch import hlfb
from ai_edge_torch.generative.layers import model_config


@dataclass
class KVCacheEntry:
  """A single cache entry include K and V caches.

  Returns:
      _type_: _description_
  """

  k_cache: torch.Tensor
  v_cache: torch.Tensor

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
  ) -> "KVCacheEntry":
    shape = (
        1,
        config.kv_cache_max,
        config.attn_config.num_query_groups,
        config.head_dim,
    )
    k = torch.zeros(shape, dtype=dtype, device=device)
    v = torch.zeros(shape, dtype=dtype, device=device)
    obj = cls(k_cache=k, v_cache=v)
    return obj


@dataclass
class KVCache:
  """A utility class for holding KV cach entries per layer."""

  caches: Tuple[KVCacheEntry]

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
  ) -> "KVCache":
    """Build an instance of the class based on model config.
    Args:
        config (ModelConfig): Model config used for building the cache.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.
    Returns:
        KVCache: The created cache object.
    """
    caches = [
        KVCacheEntry.from_model_config(config, dtype, device)
        for _ in range(config.num_layers)
    ]
    obj = cls(caches=tuple(caches))
    return obj


torch.export.register_dataclass(KVCacheEntry, serialized_type_name="kvc_entry")
torch.export.register_dataclass(KVCache, serialized_type_name="kvc_container")


def update(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
    use_hlfb: bool = True,
) -> KVCacheEntry:
  """Out of place update of Cach buffer.

  Args:
      cache (KVCacheEntry): The original cache buffer.
      input_pos (torch.Tensor): The update slice positions.
      k_slice (torch.Tensor): The K slice to be updated in the new cache.
      v_slice (torch.Tensor): The V slice to be updated in the new cache.
      use_hlfb (bool, optional): Whether the op is annotated for export.
        Defaults to True.

  Returns:
      KVCacheEntry: The updated KVCache entry based on the passed inputs.
  """
  update_func = _update_kv_hlfb_impl if use_hlfb else _update_kv_base_impl
  return update_func(cache, input_pos, k_slice, v_slice)


def _update_kv_base_impl(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntry:
  k = cache.k_cache.index_copy(1, input_pos, k_slice)
  v = cache.v_cache.index_copy(1, input_pos, v_slice)
  updated_cach = KVCacheEntry(k, v)
  return updated_cach


def _update_kv_hlfb_impl(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  builder = hlfb.StableHLOCompositeBuilder(name="odml.update_external_kv_cache")
  k_cache, v_cache, input_pos, k_slice, v_slice = builder.mark_inputs(
      cache.k_cache, cache.v_cache, input_pos, k_slice, v_slice
  )
  k = k_cache.index_copy(1, input_pos, k_slice)
  v = v_cache.index_copy(1, input_pos, v_slice)
  k, v = builder.mark_outputs(k, v)
  return KVCacheEntry(k, v)
