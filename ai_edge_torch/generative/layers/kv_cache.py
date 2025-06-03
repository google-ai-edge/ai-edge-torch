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

"""Utility functions for externalized KV Cache."""

import dataclasses
from typing import Any, List, Tuple

import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils
from ai_edge_torch.generative.layers import model_config
from ai_edge_torch.generative.utilities import types
import torch
import torch.utils._pytree as pytree


KVLayout = Tuple[types.TensorDimensionMeta, types.TensorDimensionMeta]

# Define common layouts for KV Cache.
KV_LAYOUT_DEFAULT = (types.BTNH, types.BTNH)
KV_LAYOUT_TRANSPOSED = (types.BNTH, types.BNHT)


@dataclasses.dataclass
class KVCacheEntry:
  """A single cache entry that includes K and V caches.

  The cache layout can be customized based on different use cases.
  """

  k_cache: torch.Tensor
  v_cache: torch.Tensor
  kv_layout: KVLayout = KV_LAYOUT_DEFAULT

  @classmethod
  def construct_kv_shape_from_layout(
      cls,
      shape_spec: types.TensorDimensionMeta,
      kv_cache_max: int,
      config: model_config.AttentionConfig,
      batch_size: int,
  ) -> List[int]:
    """Construct the shape of KV cache entry based on the specified layout."""
    output_shape = []
    for dim_spec in shape_spec:
      if dim_spec is types.TensorDims.BATCH:
        output_shape.append(batch_size)
      elif dim_spec is types.TensorDims.SEQUENCE:
        output_shape.append(kv_cache_max)
      elif dim_spec is types.TensorDims.NUM_HEADS:
        output_shape.append(config.num_query_groups)
      elif dim_spec is types.TensorDims.HEAD_DIM:
        output_shape.append(config.head_dim)
      else:
        raise ValueError(f"Unsupported dimension spec: {dim_spec}")
    return output_shape

  @classmethod
  def from_model_config(
      cls,
      kv_cache_max: int,
      config: model_config.AttentionConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device | None = None,
      batch_size: int = 1,
      kv_layout: KVLayout = KV_LAYOUT_DEFAULT,
  ) -> "KVCacheEntry":
    """Build an instance of the class based on model config."""
    k_shape = cls.construct_kv_shape_from_layout(
        kv_layout[0], kv_cache_max, config, batch_size
    )
    v_shape = cls.construct_kv_shape_from_layout(
        kv_layout[1], kv_cache_max, config, batch_size
    )
    k = torch.zeros(k_shape, dtype=dtype, device=device)
    v = torch.zeros(v_shape, dtype=dtype, device=device)
    obj = cls(k_cache=k, v_cache=v, kv_layout=kv_layout)
    return obj

  def get_max_seq_len(self) -> int:
    """Get the maximum sequence length in the KV cache."""
    return self.k_cache.size(
        self.kv_layout[0].dimensions.index(types.TensorDims.SEQUENCE)
    )


@dataclasses.dataclass
class KVCache:
  """A utility class for holding KV cache entries per layer."""

  caches: Tuple[KVCacheEntry, ...]

  @classmethod
  def from_model_config(
      cls,
      kv_cache_max: int,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device | None = None,
      batch_size: int = 1,
      kv_layout: KVLayout = KV_LAYOUT_DEFAULT,
  ) -> "KVCache":
    """Build an instance of the class based on model config.

    Args:
        kv_cache_max (int): The maximum sequence length in the KV cache.
        config (ModelConfig): Model config used for building the cache.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.
        batch_size (int, optional): The batch size of the cache tensors.
          Defaults to 1.

    Returns:
        KVCache: The created cache object.
    """
    caches = [
        KVCacheEntry.from_model_config(
            kv_cache_max
            if not config.block_config(idx).kv_cache_max_len
            else config.block_config(idx).kv_cache_max_len,
            config.block_config(idx).attn_config,
            dtype,
            device,
            batch_size,
            kv_layout,
        )
        for idx in range(config.num_layers)
    ]
    obj = cls(caches=tuple(caches))
    return obj

  def flatten(self) -> List[torch.Tensor]:
    """Flatten the cache entries into a list of tensors with order k_i, v_i."""
    flattened, _ = _flatten_kvc(self)
    return flattened

  def get_max_seq_len(self) -> int:
    """Get the maximum sequence length in the KV cache."""
    return self.caches[0].get_max_seq_len()


def _flatten_kvc(kvc: KVCache) -> Tuple[List[str], List[str]]:
  flattened = []
  flat_names = []
  none_names = [kvc.caches[0].kv_layout]
  for i, kv_entry in enumerate(kvc.caches):
    flattened.append(kv_entry.k_cache)
    flat_names.append(f"k_{i}")
    flattened.append(kv_entry.v_cache)
    flat_names.append(f"v_{i}")
  return flattened, [flat_names, none_names]


def _flatten_kvc_with_keys(kvc: KVCache) -> Tuple[List, List]:
  flattened, (flat_names, none_names) = _flatten_kvc(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


def _unflatten_kvc(
    values: List[torch.Tensor],
    context: Tuple[List, List],
) -> KVCache:
  assert len(values) % 2 == 0, "Found odd number of K and V entries."
  num_layers = len(values) // 2
  flat_names = context[0]
  kv_layout = context[1][0]
  kv_entries = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    v_cache_idx = flat_names.index(f"v_{i}")
    kv_entries.append(
        KVCacheEntry(
            k_cache=values[k_cache_idx],
            v_cache=values[v_cache_idx],
            kv_layout=kv_layout,
        )
    )
  obj = KVCache(tuple(kv_entries))
  return obj


def _flatten_kv_entry(
    kv_e: KVCacheEntry,
) -> Tuple[List[torch.Tensor], Any]:
  return ([kv_e.k_cache, kv_e.v_cache], kv_e.kv_layout)


def _unflatten_kv_entry(
    values: List[torch.Tensor],
    context: Any,
) -> KVCacheEntry:
  return KVCacheEntry(*values, kv_layout=context)


pytree.register_pytree_node(
    KVCacheEntry,
    _flatten_kv_entry,
    _unflatten_kv_entry,
    serialized_type_name="",
)

pytree.register_pytree_node(
    KVCache,
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
    use_dus: bool = True,
) -> KVCacheEntry:
  """Out of place update of Cache buffer.

  Args:
      cache (KVCacheEntry): The original cache buffer.
      input_pos (torch.Tensor): The update slice positions.
      k_slice (torch.Tensor): The K slice to be updated in the new cache.
      v_slice (torch.Tensor): The V slice to be updated in the new cache.

  Returns:
      KVCacheEntry: The updated KVCache entry based on the passed inputs.
  """
  update_kv_cache = _update_kv_impl if use_dus else _update_kv_base_impl
  return update_kv_cache(cache, input_pos, k_slice, v_slice)


def _update_kv_base_impl(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntry:
  """Update the cache buffer without High Level Function Boundary annotation."""
  k = cache.k_cache.index_copy(1, input_pos.to(torch.long), k_slice)
  v = cache.v_cache.index_copy(1, input_pos.to(torch.long), v_slice)
  updated_cache = KVCacheEntry(k, v)
  return updated_cache


def _get_slice_indices(positions: torch.Tensor) -> torch.Tensor:
  """Dynamic Update Slice updates are a variadic sequence of 0-rank tensors."""

  zero = torch.zeros([]).int()
  positions = positions.int()[0].reshape([])
  return [zero, positions, zero, zero]


def _update_kv_impl(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntry:
  """Update the cache buffer for K and V caches."""
  # NB: Here assume that input_pos == range(input_pos[0], len(input_pos))

  k_slice_indices = _get_slice_indices(input_pos)
  v_slice_indices = _get_slice_indices(input_pos)

  k = dus_utils.dynamic_update_slice(cache.k_cache, k_slice, k_slice_indices)
  v = dus_utils.dynamic_update_slice(cache.v_cache, v_slice, v_slice_indices)

  updated_cache = KVCacheEntry(k, v, cache.kv_layout)
  return updated_cache


def update_transposed(
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntry:
  """Out of place update of Cache buffer.

  Args:
      cache (KVCacheEntry): The original cache buffer.
      input_pos (torch.Tensor): The update slice positions.
      k_slice (torch.Tensor): The K slice to be updated in the new cache.
      v_slice (torch.Tensor): The V slice to be updated in the new cache.

  Returns:
      KVCacheEntry: The updated KVCacheBase entry based on the passed
      inputs.
  """
  assert (
      cache.kv_layout == KV_LAYOUT_TRANSPOSED
  ), "KV entry must have transposed layout."
  return _update_kv_impl_transposed(cache, input_pos, k_slice, v_slice)


def _get_slice_indices_transposed(
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
    cache: KVCacheEntry,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntry:
  """Updates the cache buffer with High Level Function Boundary annotation."""
  cache_dim = 4
  k_ts_idx = 2
  v_ts_idx = 3
  positions = input_pos.clone()
  k_slice_indices = _get_slice_indices_transposed(
      positions, cache_dim, k_ts_idx
  )
  v_slice_indices = _get_slice_indices_transposed(
      positions, cache_dim, v_ts_idx
  )
  k = dus_utils.dynamic_update_slice(
      cache.k_cache, k_slice, [x for x in k_slice_indices]
  )
  v = dus_utils.dynamic_update_slice(
      cache.v_cache, v_slice, [x for x in v_slice_indices]
  )
  return KVCacheEntry(k, v, cache.kv_layout)
