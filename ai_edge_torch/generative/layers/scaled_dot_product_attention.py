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
# Implements scaled dot product attention.

import math
from typing import Optional

from ai_edge_torch.generative.custom_ops import bmm_4d as bmm_lib
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.hlfb import StableHLOCompositeBuilder
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    alibi_bias: Optional[torch.Tensor] = None,
):
  """Scaled dot product attention.

  Args:
    q (torch.Tensor): Query tensor, with shape [B, T, N, H].
    k (torch.Tensor): Key tensor, with shape [B, T, KV_LEN, H].
    v (torch.Tensor): Value tensor, with shape [B, T, KV_LEN, H].
    head_size (int): head dimension.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.
    alibi_bias (torch.Tensor): optional alibi bias tensor.

  Returns:
    The output tensor of scaled_dot_product_attention.
  """
  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  if alibi_bias is not None:
    alibi_bias = alibi_bias * scale
    if mask is None:
      mask = alibi_bias
    else:
      mask = mask + alibi_bias

  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  if q.size() != k.size():
    # Handle the GQA case, where q.shape[1] % k.shape[1] == 0.
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
  if softcap is None:
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=mask is None,
        scale=scale,
    )
  else:
    q = q * scale
    scores = q @ k.transpose(-1, -2)
    scores = scores / softcap
    scores = torch.tanh(scores)
    scores = scores * softcap
    if mask is not None:
      scores = scores + mask
    out = F.softmax(scores.float(), dim=-1).type_as(q)
    y = torch.matmul(out, v)

  return y.transpose(1, 2)


def scaled_dot_product_attention_with_hlfb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    alibi_bias: Optional[torch.Tensor] = None,
):
  """Scaled dot product attention with high-level function boundary enabled.

  Args:
    q (torch.Tensor): Query tensor, with shape [B, T, N, H].
    k (torch.Tensor): Key tensor, with shape [B, T, KV_LEN, H].
    v (torch.Tensor): Value tensor, with shape [B, T, KV_LEN, H].
    head_size (int): head dimension.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.
    alibi_bias (torch.Tensor): optional alibi bias tensor.

  Returns:
    The output tensor of scaled_dot_product_attention.
  """
  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  if alibi_bias is not None:
    alibi_bias = alibi_bias * scale
    if mask is None:
      mask = alibi_bias
    else:
      mask = mask + alibi_bias

  attrs = {"scale": scale}

  if softcap is not None:
    attrs["logit_cap"] = softcap

  builder = StableHLOCompositeBuilder(
      name="odml.scaled_dot_product_attention", attr=attrs
  )
  q, k, v, mask = builder.mark_inputs(q, k, v, mask)

  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  if q.size() != k.size():
    # Handle the GQA case, where q.shape[1] % k.shape[1] == 0.
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
  if softcap is None:
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=mask is None,
        scale=scale,
    )
  else:
    q = q * scale
    scores = q @ k.transpose(-1, -2)
    scores = scores / softcap
    scores = torch.tanh(scores)
    scores = scores * softcap
    if mask is not None:
      scores = scores + mask
    out = F.softmax(scores.float(), dim=-1).type_as(q)
    y = torch.matmul(out, v)

  result = y.transpose(1, 2)
  result = builder.mark_outputs(result)
  return result


def scaled_dot_product_attention_transposed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    alibi_bias: Optional[torch.Tensor] = None,
):
  """Scaled dot product attention with transposed key and value.

  Args:
    query: Query tensor, with shape [B, T, N, H].
    key: Key tensor, with shape [B, T, KV_LEN, H].
    value: Value tensor, with shape [B, T, H, KV_LEN].
    head_size (int): head dimension.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.
    alibi_bias (torch.Tensor): optional alibi bias tensor.

  Returns:
    The output tensor of scaled_dot_product_attention_transposed.
  """
  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  if alibi_bias is not None:
    alibi_bias = alibi_bias * scale
    if mask is None:
      mask = alibi_bias
    else:
      mask = mask + alibi_bias

  query = query * scale

  assert mask is not None, "Mask should not be None!"
  t = mask.shape[2]

  logits = bmm_lib.bmm_4d(query, key)

  _, bk, gt, s = logits.shape
  g = gt // t
  logits = logits.reshape((bk, g, t, s))
  if softcap is not None:
    logits = torch.tanh(logits / softcap)
    logits = logits * softcap

  padded_logits = logits + mask
  padded_logits = padded_logits.reshape(1, bk, gt, s)
  probs = F.softmax(padded_logits, dim=-1).type_as(key)
  encoded = bmm_lib.bmm_4d(probs, value)

  return encoded  # 1, bk, gt, h
