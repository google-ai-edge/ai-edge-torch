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
# Implementation for Rotary Position embedding. https://arxiv.org/pdf/2104.09864.pdf
from typing import Tuple
import torch


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
  """Computes rotary positional embedding.

  Args:
    x: the input tensor.
    cos: cosine value for the rope.
    sin: sin value for the rope.

  Returns:
    output tensor of RoPE.
  """
  x = x.transpose(1, 2)
  head_size = x.size(-1)
  x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
  x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
  rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
  roped = (x * cos) + (rotated * sin)
  return roped.transpose(1, 2).type_as(x)


def apply_rope_inline(
    q: torch.Tensor,
    k: torch.Tensor,
    input_pos: torch.Tensor,
    n_elem: int,
    base: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes rotary positional embedding inline for a query and key.

  Args:
    q: the query tensor.
    k: the key tensor.
    input_pos: the sequence indices for the query and key
    n_elem: number of elements of the head dimension for RoPE computation

  Returns:
    output the RoPE'd query and key.
  """

  if n_elem <= 0:
    return q, k

  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2).float() / n_elem))
  freq_exponents = (2.0 / n_elem) * torch.arange(
      q.shape[-1] // 2, dtype=torch.float32
  )
  timescale = float(base) ** freq_exponents
  radians = input_pos.clone().unsqueeze(0).unsqueeze(-1) / timescale.unsqueeze(
      0
  ).unsqueeze(0)
  cos = torch.cos(radians).type_as(q)
  sin = torch.sin(radians).type_as(q)

  def apply(x, sin, cos):
    x = x.transpose(1, 2)
    b, h, s, d = x.shape
    ans = torch.split(x, d // 2, dim=-1)
    x1, x2 = ans
    left = x1 * cos - x2 * sin
    right = x2 * cos + x1 * sin
    res = torch.cat([left, right], dim=-1)
    res = res.transpose(1, 2)
    return res

  q_roped = apply(q, sin, cos)
  k_roped = apply(k, sin, cos)
  return q_roped, k_roped
