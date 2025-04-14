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
# Implements scaled dot product attention.  This is experimental and
# GPU-specific code.

import math
from typing import Optional

from ai_edge_torch.generative.custom_ops import bmm_4d as bmm_lib
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.utilities import types
from ai_edge_torch.hlfb import StableHLOCompositeBuilder
from multipledispatch import dispatch
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    kv: kv_utils.KVCacheEntry,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
):
  if hasattr(kv, "kv_layout"):
    return _sdpa(
        kv.kv_layout[0](),  # key layout
        kv.kv_layout[1](),  # value layout
        query=query,
        key=key,
        value=value,
        head_size=head_size,
        mask=mask,
        scale=scale,
        softcap=softcap,
    )
  raise ValueError("No kv_layout attribute found in kv.")


@dispatch(types.BNTH, types.BNHT)
def _sdpa(k_type, v_type, *args, **kwargs):
  query = kwargs["query"]
  key = kwargs["key"]
  value = kwargs["value"]
  head_size = kwargs["head_size"]
  mask = kwargs.get("mask", None)
  scale = kwargs.get("scale", None)
  softcap = kwargs.get("softcap", None)

  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

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


@dispatch(object, object)
def _sdpa(k_type, v_type, *args, **kwargs):

  raise ValueError(f"No implementations for k={k_type} and v={v_type}")
