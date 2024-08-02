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
from typing import List, Tuple

from ai_edge_torch import hlfb
from ai_edge_torch.generative.layers import model_config
import torch
import torch.utils._pytree as pytree


@dataclass
class KVCacheEntry:
  """A single cache entry include K and V caches."""

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
        config.attn_config.head_dim,
    )
    k = torch.zeros(shape, dtype=dtype, device=device)
    v = torch.zeros(shape, dtype=dtype, device=device)
    obj = cls(k_cache=k, v_cache=v)
    return obj


@dataclass
class EKVCache:
  """A utility class for holding KV cache entries per layer."""

  caches: Tuple[KVCacheEntry]

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
  ) -> "EKVCache":
    """Build an instance of the class based on model config.
    Args:
        config (ModelConfig): Model config used for building the cache.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.
    Returns:
        EKVCache: The created cache object.
    """
    caches = [
        KVCacheEntry.from_model_config(config, dtype, device)
        for _ in range(config.num_layers)
    ]
    obj = cls(caches=tuple(caches))
    return obj


def _flatten_kvc(kvc: EKVCache) -> Tuple[List, List]:
  flattened = []
  flat_names = []
  none_names = []
  for i, kv_entry in enumerate(kvc.caches):
    flattened.append(kv_entry.k_cache)
    flat_names.append(f"k_{i}")
    flattened.append(kv_entry.v_cache)
    flat_names.append(f"v_{i}")
  return flattened, [flat_names, none_names]


def _flatten_kvc_with_keys(kvc: EKVCache) -> Tuple[List, List]:
  flattened, (flat_names, none_names) = _flatten_kvc(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


def _unflatten_kvc(
    values: List[torch.Tensor], context: Tuple[List, List]
) -> EKVCache:
  assert len(values) % 2 == 0, "Found odd number of K and V entries."
  num_layers = len(values) // 2
  flat_names = context[0]
  kv_entries = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    v_cache_idx = flat_names.index(f"v_{i}")
    kv_entries.append(
        KVCacheEntry(k_cache=values[k_cache_idx], v_cache=values[v_cache_idx])
    )
  obj = EKVCache(tuple(kv_entries))
  return obj


pytree.register_pytree_node(
    EKVCache,
    _flatten_kvc,
    _unflatten_kvc,
    flatten_with_keys_fn=_flatten_kvc_with_keys,
    serialized_type_name="",
)


def update(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
    use_hlfb: bool = True,
) -> KVCacheEntry:
  """Out of place update of Cache buffer.

  Args:
      cache (KVCacheEntry): The original cache buffer.
      input_pos (torch.Tensor): The update slice positions.
      k_slice (torch.Tensor): The K slice to be updated in the new cache.
      v_slice (torch.Tensor): The V slice to be updated in the new cache.
      use_hlfb (bool, optional): Whether the op is annotated for export with
        High Level Function Boundary. Defaults to True.

  Returns:
      KVCacheEntry: The updated EKVCache entry based on the passed inputs.
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
  updated_cache = KVCacheEntry(k, v)
  return updated_cache


def _update_kv_hlfb_impl(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntry:
  builder = hlfb.StableHLOCompositeBuilder(name="odml.update_external_kv_cache")
  k_cache, v_cache, input_pos, k_slice, v_slice = builder.mark_inputs(
      cache.k_cache, cache.v_cache, input_pos, k_slice, v_slice
  )
  k = k_cache.index_copy(1, input_pos, k_slice)
  v = v_cache.index_copy(1, input_pos, v_slice)
  k, v = builder.mark_outputs(k, v)
  return KVCacheEntry(k, v)
