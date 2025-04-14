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
# Common utility functions for data loading etc.
from dataclasses import dataclass
from typing import Tuple
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import scaled_dot_product_attention as sdpa_default
from ai_edge_torch.generative.layers.experimental import kv_cache as kv_utils_experimental
from ai_edge_torch.generative.layers.experimental import scaled_dot_product_attention as sdpa
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import types
from multipledispatch import dispatch
import torch


def sdpa_with_kv_update(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv: kv_utils.KVCacheEntry,
    input_pos: torch.Tensor,
    mask: torch.Tensor,
    config: cfg.AttentionConfig,
) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
  return sdpa_with_kv_update_impl(
      kv.kv_layout[0](),  # key layout
      kv.kv_layout[1](),  # value layout
      query=query,
      key=key,
      value=value,
      kv=kv,
      input_pos=input_pos,
      mask=mask,
      config=config,
  )


@dispatch(types.BNTH, types.BNHT)
def sdpa_with_kv_update_impl(
    k_type, v_type, *args, **kwargs
) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
  query = kwargs["query"]
  key = kwargs["key"]
  value = kwargs["value"]
  kv = kwargs["kv"]
  input_pos = kwargs["input_pos"]
  mask = kwargs["mask"]
  config = kwargs["config"]

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

  if kv is not None:
    kv = kv_utils_experimental.update(kv, input_pos, key, value)
    key, value = kv.k_cache, kv.v_cache

  sdpa_out = sdpa.scaled_dot_product_attention(
      kv,
      query,
      key,
      value,
      config.head_dim,
      mask=mask,
      softcap=config.logit_softcap,
  )  # 1, bk, gt, h
  sdpa_out = (
      sdpa_out.reshape(b, -1, seq_len, h)
      .permute(0, 2, 1, 3)
      .reshape(b, seq_len, -1)
  )
  return sdpa_out, kv


@dispatch(object, object)
def sdpa_with_kv_update_impl(
    k_type, v_type, *args, **kwargs
) -> Tuple[torch.Tensor, kv_utils.KVCacheEntry]:
  query = kwargs["query"]
  key = kwargs["key"]
  value = kwargs["value"]
  kv = kwargs["kv"]
  input_pos = kwargs["input_pos"]
  mask = kwargs["mask"]
  config = kwargs["config"]

  b, seq_len, _, _ = query.shape
  if kv is not None:
    kv = kv_utils.update(kv, input_pos, key, value)
    key, value = kv.k_cache, kv.v_cache

  sdpa_out = sdpa_default.scaled_dot_product_attention(
      query,
      key,
      value,
      config.head_dim,
      mask=mask,
      softcap=config.logit_softcap,
  )
  sdpa_out = sdpa_out.reshape(b, seq_len, -1)
  return sdpa_out, kv
