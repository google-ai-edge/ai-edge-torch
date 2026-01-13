# Copyright 2026 The AI Edge Torch Authors.
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
"""Optimized Cache class for HuggingFace integration.

Shape annotations used here:
  B: batch size
  K: num_key_value_heads
  G: number of KV groups
  N: number of attention heads. N // K = G
  T: target / input length
  S: sequence / context length
  H: head dimension
"""

from typing import Any, List, Optional, Self, Tuple
import ai_edge_torch.generative.export_hf.core.cache_base as cache_base_lib
import jaxtyping as jt
import torch
import torch.utils._pytree as pytree


KeyCache = jt.Shaped[torch.Tensor, "1 BK H S"]
KeySlice = jt.Shaped[torch.Tensor, "1 BK H T"]
ValueCache = jt.Shaped[torch.Tensor, "1 BK S H"]
ValueSlice = jt.Shaped[torch.Tensor, "1 BK T H"]

KeyCacheEntry = Tuple[KeyCache, KeySlice | None]
ValueCacheEntry = Tuple[ValueCache, ValueSlice | None]


class LiteRTLMSplitCacheLayer(cache_base_lib.LiteRTLMCacheLayerMixin):
  """Optimized Cache layer class for HuggingFace integration."""

  is_compileable = True
  is_sliding = False

  def __init__(
      self,
      key_cache: KeyCacheEntry,
      value_cache: ValueCacheEntry,
      batch_size: int = 1,
      **kwargs,
  ):
    super().__init__()
    if not isinstance(key_cache, tuple):
      key_cache = (key_cache, None)
    if not isinstance(value_cache, tuple):
      value_cache = (value_cache, None)
    self.keys = key_cache
    self.values = value_cache
    self.is_initialized = True

    self.k_cache_shape = self.keys[0].shape
    self.v_cache_shape = self.values[0].shape
    self.max_cache_len = self.k_cache_shape[3]
    self.batch_size = batch_size
    self.num_key_value_heads = self.k_cache_shape[1] // self.batch_size
    self.head_dim = self.k_cache_shape[2]
    self.additional_states = kwargs.get("additional_states", None)

    self.cumulative_length = 0

  def get_batch_size(self) -> int:
    return self.batch_size

  def lazy_initialization(self, key_states: torch.Tensor):
    # Since we don't support real lazy initialization, this function could only
    # be called by Cache.early_initialization, where uses a standard cache
    # layout [batch_size, num_heads, ?, head_dim].
    # TODO(weiyiw): Implement this function.
    raise NotImplementedError(
        "Lazy initialization is not supported in LiteRTLMCacheLayer."
    )

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      cache_kwargs: Optional[dict[str, Any]] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = key_states.shape[2]
    self.cumulative_length += seq_len

    key_states = key_states.to(self.keys[0].dtype)

    value_states = value_states.to(self.values[0].dtype)

    key_states = key_states.permute(0, 1, 3, 2).reshape(
        1, -1, self.head_dim, seq_len
    )  # 1, bk, h, s
    value_states = value_states.reshape(
        1, -1, seq_len, self.head_dim
    )  # 1, bk, s, h

    self.keys = (self.keys[0], key_states)
    self.values = (self.values[0], value_states)

    return self.keys, self.values

  def get_mask_sizes(self, cache_position: torch.Tensor):
    """Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for."""
    kv_offset = 0
    kv_length = self.max_cache_len
    return kv_length, kv_offset

  def get_seq_length(self) -> int:
    return (self.keys[0][0, 0].any(dim=-1)).sum() if self.is_initialized else 0

  def get_max_cache_shape(self) -> int:
    return self.max_cache_len

  @classmethod
  def _infer_cache_shape_from_config(
      cls, model_config, layer_index, cache_length, batch_size=1
  ):
    """Infers the KV cache shape from the model config."""
    del layer_index  # Unused.
    num_kv_heads = model_config.num_key_value_heads
    embed_size_per_head = (
        getattr(model_config, "head_dim", None)
        or model_config.hidden_size // model_config.num_attention_heads
    )

    k_cache_shape = (
        1,
        batch_size * num_kv_heads,
        embed_size_per_head,
        cache_length,
    )
    v_cache_shape = (
        1,
        batch_size * num_kv_heads,
        cache_length,
        embed_size_per_head,
    )
    return k_cache_shape, v_cache_shape

  @classmethod
  def create_from_config(
      cls, model_config, layer_index, cache_length, batch_size=1, **kwargs
  ) -> Self:
    """Creates a KV cache from the model config."""
    k_cache_shape, v_cache_shape = cls._infer_cache_shape_from_config(
        model_config, layer_index, cache_length, batch_size
    )
    keys = torch.zeros(k_cache_shape, dtype=torch.float32)
    values = torch.zeros(v_cache_shape, dtype=torch.float32)
    return cls((keys, None), (values, None), **kwargs)


@cache_base_lib.register_cache_implementation
class LiteRTLMSplitCache(cache_base_lib.LiteRTLMCacheMixin):
  """Optimized Cache class for HuggingFace integration."""

  @classmethod
  def create_from_config(cls, model_config, cache_length, batch_size=1) -> Self:
    """Creates a KV cache from the model config."""
    num_layers = model_config.num_hidden_layers
    layers = []
    for layer_index in range(num_layers):
      layers.append(
          LiteRTLMSplitCacheLayer.create_from_config(
              model_config, layer_index, cache_length, batch_size=batch_size
          )
      )
    return cls(layers)


def _flatten_kvc_t(
    kvc: LiteRTLMSplitCache,
) -> Tuple[List[torch.Tensor], Tuple[List[str], Tuple[int, int]]]:
  """Flattens the KV cache to a list of tensors."""
  flattened = []
  flat_names = []
  num_layers = len(kvc.layers)
  layer_0 = kvc.layers[0]
  assert isinstance(layer_0, cache_base_lib.LiteRTLMCacheLayerMixin)
  batch_size = layer_0.get_batch_size()
  for i, cache_layer in enumerate(kvc.layers):
    flattened.append(cache_layer.keys[0])
    flat_names.append(f"k_{i}")
    flattened.append(cache_layer.values[0])
    flat_names.append(f"v_{i}")
    if cache_layer.keys[1] is not None:
      assert cache_layer.values[1] is not None
      flattened.append(cache_layer.keys[1])
      flat_names.append(f"k_{i}_slice")
      flattened.append(cache_layer.values[1])
      flat_names.append(f"v_{i}_slice")
  return flattened, [flat_names, (batch_size, num_layers)]


def _unflatten_kvc_t(
    values: List[torch.Tensor],
    context: Tuple[List[str], Tuple[int, int]],
) -> LiteRTLMSplitCache:
  """Unflattens the KV cache from a list of tensors."""
  flat_names = context[0]
  batch_size = context[1][0]
  num_layers = context[1][1]
  kv_entries = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    k_cache = values[k_cache_idx]
    try:
      k_slice_idx = flat_names.index(f"k_{i}_slice")
      k_cache_update = values[k_slice_idx]
    except ValueError:
      k_cache_update = None
    v_cache_idx = flat_names.index(f"v_{i}")
    v_cache = values[v_cache_idx]
    try:
      v_slice_idx = flat_names.index(f"v_{i}_slice")
      v_cache_update = values[v_slice_idx]
    except ValueError:
      v_cache_update = None
    kv_entries.append(
        LiteRTLMSplitCacheLayer(
            key_cache=(k_cache, k_cache_update),
            value_cache=(v_cache, v_cache_update),
            batch_size=batch_size,
        )
    )
  obj = LiteRTLMSplitCache(kv_entries)
  return obj


def _flatten_kvc_t_with_keys(
    kvc: LiteRTLMSplitCache,
):
  flattened, (flat_names, _) = _flatten_kvc_t(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    LiteRTLMSplitCache,
    _flatten_kvc_t,
    _unflatten_kvc_t,
    flatten_with_keys_fn=_flatten_kvc_t_with_keys,
    serialized_type_name="",
)
