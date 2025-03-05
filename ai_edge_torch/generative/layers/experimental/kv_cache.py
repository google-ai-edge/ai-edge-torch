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

import dataclasses
import functools
from typing import Any, List, Tuple, Type
from ai_edge_torch.generative.layers import model_config
from ai_edge_torch.generative.layers.experimental import types
from ai_edge_torch.generative.utilities import dynamic_update_slice as dus_utils
import torch
import torch.utils._pytree as pytree


@dataclasses.dataclass
class KVCacheEntryBase:
  """A single cache entry that includes K and V caches.

  The chaches are built based on the provided config with the shape of
  (batch_size, kv_cache_max, num_query_groups, head_dim).
  """

  k_cache: torch.Tensor
  v_cache: torch.Tensor

  @classmethod
  def _from_model_config(
      cls,
      k_shape: Tuple[int, ...],
      v_shape: Tuple[int, ...],
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
  ):
    """Build an instance of the class based on model config."""
    k = torch.zeros(k_shape, dtype=dtype, device=device)
    v = torch.zeros(v_shape, dtype=dtype, device=device)
    obj = cls(k_cache=k, v_cache=v)
    return obj

  @classmethod
  def from_model_config(
      cls,
      kv_cache_max: int,
      config: model_config.AttentionConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
      batch_size: int = 1,
  ):
    """Build an instance of the class based on model config."""
    shape = (batch_size, kv_cache_max, config.num_query_groups, config.head_dim)
    return cls._from_model_config(shape, shape, dtype, device)


@dataclasses.dataclass
class KVCacheEntryBTNH(KVCacheEntryBase):
  k_type = types.BTNH()
  v_type = types.BTNH()


@dataclasses.dataclass
class KVCacheEntryTransposed(KVCacheEntryBase):

  k_type = types.BNTH()
  v_type = types.BNHT()

  @classmethod
  def from_model_config(
      cls,
      kv_cache_max: int,
      config: model_config.AttentionConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
      batch_size: int = 1,
  ):
    """Build an instance of the class based on model config."""
    k_shape = (
        batch_size,
        config.num_query_groups,
        kv_cache_max,
        config.head_dim,
    )  # b, k, s, h
    v_shape = (
        batch_size,
        config.num_query_groups,
        config.head_dim,
        kv_cache_max,
    )  # b, k, h, s
    return cls._from_model_config(k_shape, v_shape, dtype, device)


def _flatten_kv_entry(
    kv_e: KVCacheEntryBase,
) -> Tuple[List[torch.Tensor], Any]:
  return ([kv_e.k_cache, kv_e.v_cache], None)


def _unflatten_kv_entry(
    kv_entry_ty: Type[KVCacheEntryBase],
    values: List[torch.Tensor],
    unused_context: Any,
) -> KVCacheEntryBase:
  return kv_entry_ty(*values)


pytree.register_pytree_node(
    KVCacheEntryTransposed,
    _flatten_kv_entry,
    functools.partial(_unflatten_kv_entry, KVCacheEntryTransposed),
    serialized_type_name="",
)

pytree.register_pytree_node(
    KVCacheEntryBase,
    _flatten_kv_entry,
    functools.partial(_unflatten_kv_entry, KVCacheEntryBase),
    serialized_type_name="",
)


@dataclasses.dataclass
class KVCacheBase:
  """A utility class for holding KV cache entries per layer."""

  caches: Tuple[KVCacheEntryBase, ...]

  @classmethod
  def _from_model_config(
      cls,
      kv_entry_cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
      batch_size: int = 1,
  ):
    caches = [
        kv_entry_cls.from_model_config(
            config.kv_cache_max,
            config.block_config(idx).attn_config,
            dtype,
            device,
            batch_size,
        )
        for idx in range(config.num_layers)
    ]
    obj = cls(caches=tuple(caches))
    return obj

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
      batch_size: int = 1,
  ):
    """Build an instance of the class based on model config.

    Args:
        config (ModelConfig): Model config used for building the cache.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.
        batch_size (int, optional): The batch size of the cache tensors.
          Defaults to 1.

    Returns:
        KVCacheBase: The created cache object.
    """
    assert batch_size == 1, "Batch size must be 1 for KV Cache."
    return cls._from_model_config(
        KVCacheEntryBase,
        config=config,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
    )

  def flatten(self) -> List[torch.Tensor]:
    """Flatten the cache entries into a list of tensors with order k_i, v_i."""
    flattened, _ = _flatten_kvc(self)
    return flattened


@dataclasses.dataclass
class KVCacheBTNH(KVCacheBase):

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
      batch_size: int = 1,
  ):
    return cls._from_model_config(
        KVCacheEntryBTNH,
        config=config,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
    )


@dataclasses.dataclass
class KVCacheTransposed(KVCacheBase):

  @classmethod
  def from_model_config(
      cls,
      config: model_config.ModelConfig,
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
      batch_size: int = 1,
  ):
    return cls._from_model_config(
        KVCacheEntryTransposed,
        config=config,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
    )


def _flatten_kvc(kvc: KVCacheBase) -> Tuple[List[str], List[str]]:
  flattened = []
  flat_names = []
  none_names = []
  for i, kv_entry in enumerate(kvc.caches):
    flattened.append(kv_entry.k_cache)
    flat_names.append(f"k_{i}")
    flattened.append(kv_entry.v_cache)
    flat_names.append(f"v_{i}")
  return flattened, [flat_names, none_names]


def _flatten_kvc_with_keys(kvc: KVCacheBase) -> Tuple[List, List]:
  flattened, (flat_names, none_names) = _flatten_kvc(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


def _unflatten_kvc(
    kv_ty: Type[KVCacheBase],
    kv_entry_type: Type[KVCacheEntryBase],
    values: List[torch.Tensor],
    context: Tuple[List, List],
) -> KVCacheBase:
  assert len(values) % 2 == 0, "Found odd number of K and V entries."
  num_layers = len(values) // 2
  flat_names = context[0]
  kv_entries = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    v_cache_idx = flat_names.index(f"v_{i}")
    kv_entries.append(
        kv_entry_type(k_cache=values[k_cache_idx], v_cache=values[v_cache_idx])
    )
  obj = kv_ty(tuple(kv_entries))
  return obj


pytree.register_pytree_node(
    KVCacheTransposed,
    _flatten_kvc,
    functools.partial(
        _unflatten_kvc, KVCacheTransposed, KVCacheEntryTransposed
    ),
    flatten_with_keys_fn=_flatten_kvc_with_keys,
    serialized_type_name="",
)

pytree.register_pytree_node(
    KVCacheBase,
    _flatten_kvc,
    functools.partial(_unflatten_kvc, KVCacheBase, KVCacheEntryBase),
    flatten_with_keys_fn=_flatten_kvc_with_keys,
    serialized_type_name="",
)


def update(
    cache: KVCacheEntryBase,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntryBase:
  """Out of place update of Cache buffer.

  Args:
      cache (KVCacheEntryBase): The original cache buffer.
      input_pos (torch.Tensor): The update slice positions.
      k_slice (torch.Tensor): The K slice to be updated in the new cache.
      v_slice (torch.Tensor): The V slice to be updated in the new cache.

  Returns:
      KVCacheEntryBase: The updated KVCacheBase entry based on the passed
      inputs.
  """
  update_kv_cache = _update_kv_impl
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


def _update_kv_impl(
    cache: KVCacheEntryTransposed,
    input_pos: torch.Tensor,
    k_slice: torch.Tensor,
    v_slice: torch.Tensor,
) -> KVCacheEntryTransposed:
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
  return KVCacheEntryTransposed(k, v)
