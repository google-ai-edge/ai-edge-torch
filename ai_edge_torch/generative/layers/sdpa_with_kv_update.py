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

"""Common utility functions for data loading etc."""

from typing import Optional, Tuple

from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import scaled_dot_product_attention as sdpa
import ai_edge_torch.generative.layers.model_config as cfg
import torch


def sdpa_with_kv_update(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv: kv_utils.KVCacheEntry,
    input_pos: torch.Tensor,
    mask: torch.Tensor,
    config: cfg.AttentionConfig,
    enable_hlfb: bool,
    alibi_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
  """Wrapper function for scaled dot product attention with KV cache update."""
  if kv is not None and kv.kv_layout == kv_utils.KV_LAYOUT_TRANSPOSED:
    return _sdpa_with_kv_update_transposed(
        query, key, value, kv, input_pos, mask, config, alibi_bias
    )
  return _sdpa_with_kv_update_default(
      query, key, value, kv, input_pos, mask, config, enable_hlfb, alibi_bias
  )


def _sdpa_with_kv_update_transposed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv: kv_utils.KVCacheEntry,
    input_pos: torch.Tensor,
    mask: torch.Tensor,
    config: cfg.AttentionConfig,
    alibi_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
  # Transpose k/v to specific layout for GPU implementation.
  b, seq_len, n, h = query.shape
  g = n // config.num_query_groups
  # btnh -> bnth -> b(kg)th -> 1(bk)(gt)h
  query = query.permute(0, 2, 1, 3).reshape(
      1, b * config.num_query_groups, g * seq_len, h
  )

  key = key.permute(0, 2, 1, 3).reshape(
      1, -1, seq_len, config.head_dim
  )  # 1, bk, s, h
  value = value.permute(0, 2, 3, 1).reshape(
      1, -1, config.head_dim, seq_len
  )  # 1, bk, h, s

  kv = kv_utils.update_transposed(kv, input_pos, key, value)
  key, value = kv.k_cache, kv.v_cache

  sdpa_out = sdpa.scaled_dot_product_attention_transposed(
      query,
      key,
      value,
      config.head_dim,
      mask=mask,
      softcap=config.logit_softcap,
      alibi_bias=alibi_bias,
  )  # 1, bk, gt, h
  sdpa_out = (
      sdpa_out.reshape(b, -1, seq_len, h)
      .permute(0, 2, 1, 3)
      .reshape(b, seq_len, -1)
  )
  return sdpa_out, kv


def _sdpa_with_kv_update_default(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv: kv_utils.KVCacheEntry,
    input_pos: torch.Tensor,
    mask: torch.Tensor,
    config: cfg.AttentionConfig,
    enable_hlfb: bool,
    alibi_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
  b, seq_len, _, _ = query.shape
  if kv is not None:
    kv = kv_utils.update(kv, input_pos, key, value)
    key, value = kv.k_cache, kv.v_cache

  if enable_hlfb:
    sdpa_func = sdpa.scaled_dot_product_attention_with_hlfb
  else:
    sdpa_func = sdpa.scaled_dot_product_attention
  sdpa_out = sdpa_func(
      query,
      key,
      value,
      config.head_dim,
      mask=mask,
      softcap=config.logit_softcap,
      alibi_bias=alibi_bias,
  )
  sdpa_out = sdpa_out.reshape(b, seq_len, -1)
  return sdpa_out, kv
