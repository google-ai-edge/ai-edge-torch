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
"""Optimized Attention layer for HuggingFace integration."""


from ai_edge_torch.generative.layers import scaled_dot_product_attention as sdpa_lib
import jaxtyping as jt
import torch
import transformers


def transposed_attention(
    module: torch.nn.Module,
    query: jt.Float[torch.Tensor, "b n t h"],
    key: jt.Float[torch.Tensor, "1 c s h"],
    value: jt.Float[torch.Tensor, "1 c h s"],
    attention_mask: jt.Shaped[torch.Tensor, "1 1 t s"] | None,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
  """Transpose k/v to specific layout for LiteRT Optimized implementation.

  Args:
    module: The attention layer module.
    query: The query tensor.
    key: The key cache tensor. Note that the key cache tensor is pre-processed.
    value: The value tensor. Note that the key cache tensor is pre-processed.
    attention_mask: The attention mask tensor.
    scaling: The scaling factor.
    softcap: The softcap factor.
    **kwargs: Other keyword arguments.

  Returns:
    The attention output tensor.
  """
  del kwargs  # Unused in this implementation but required by the interface.

  b, n, seq_len, h = query.shape
  g = getattr(module, "num_key_value_groups", 1)
  num_query_groups = n // g
  # bnth -> b(kg)th -> 1(bk)(gt)h
  query = query.reshape(1, b * num_query_groups, g * seq_len, h)

  # 1, bk, gt, h
  sdpa_out = sdpa_lib.scaled_dot_product_attention_transposed(
      query,
      key,
      value,
      h,
      mask=attention_mask,
      scale=scaling,
      softcap=softcap,
  )
  # b, kg, t, h
  sdpa_out = sdpa_out.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
  return sdpa_out, None


transformers.AttentionInterface.register(
    "lrt_transposed_attention", transposed_attention
)
