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
"""Custom Split Cache attention implementation for ODML."""

import math
from typing import Optional

from ai_edge_torch.generative.export_hf.core.split_cache import cache as kv_cache_lib
import torch
import torch.nn.functional as F
import transformers


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key_cache: kv_cache_lib.KeyCacheEntry,
    value_cache: kv_cache_lib.ValueCacheEntry,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
):
  """Scaled dot product attention with transposed key and value.

  Args:
    query: Query tensor, 1(bk)(gt)h.
    key_cache: A tuple of Key tensor. 1(bk)hs
    value_cache: A tuple of Value tensor. 1(bk)sh
    head_size (int): head dimension.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.

  Returns:
    The output tensor of scaled_dot_product_attention_transposed.
  """
  key_past = key_cache[0]
  key = key_cache[1]

  value_past = value_cache[0]
  value = value_cache[1]

  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  query = query * scale

  assert mask is not None, "Mask should not be None!"
  t = mask.shape[2]

  logits0 = torch.einsum("abth,abhs->abts", query, key_past)
  logits1 = torch.einsum("abth,abhs->abts", query, key)
  logits = torch.cat([logits0, logits1], dim=-1)

  _, bk, gt, s = logits.shape
  g = gt // t
  logits = logits.reshape((bk, g, t, s))
  if softcap is not None:
    logits = torch.tanh(logits / softcap)
    logits = logits * softcap

  padded_logits = logits + mask
  padded_logits = padded_logits.reshape(1, bk, gt, s)
  probs = F.softmax(padded_logits, dim=-1).type_as(key)
  probs0, probs1 = probs[..., :-t], probs[..., -t:]

  encoded0 = torch.einsum("abts,absh->abth", probs0, value_past)
  encoded1 = torch.einsum("abts,absh->abth", probs1, value)
  encoded = encoded0 + encoded1

  return encoded  # 1, bk, gt, h


def split_cache_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: kv_cache_lib.KeyCacheEntry,  # required arg
    value: kv_cache_lib.ValueCacheEntry,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """ODML transposed attention implementation for NPU."""
  del kwargs

  b, n, seq_len, h = query.shape
  if hasattr(module, "num_key_value_groups"):
    g = module.num_key_value_groups
  else:
    g = 1
  num_query_groups = n // g
  # bnth -> b(kg)th -> 1(bk)(gt)h
  query = query.reshape(1, b * num_query_groups, g * seq_len, h)

  sdpa_out = _scaled_dot_product_attention(
      query,
      key,
      value,
      h,
      mask=attention_mask,
      scale=scaling,
      softcap=softcap,
  )  # 1, bk, gt, h
  sdpa_out = sdpa_out.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
  return sdpa_out, None


transformers.AttentionInterface.register(
    "lrt_split_cache_attention", split_cache_attention
)
