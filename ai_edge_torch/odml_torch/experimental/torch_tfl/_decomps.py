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
"""Torch ops to Torch-TFL decompositions."""
from ai_edge_torch.odml_torch.experimental.torch_tfl import _ops
import torch

decomps = {}


def register_decomp(op):
  global decomps
  ops = [op]
  if isinstance(op, torch._ops.OpOverloadPacket):
    ops = [getattr(op, overload) for overload in op.overloads()]

  def register(decomp_fn):
    for op in ops:
      decomps[op] = decomp_fn
    return decomp_fn

  return register


@register_decomp(torch.ops.aten.mm.default)
def _aten_mm_decomp(x, y):
  return torch.ops.tfl.batch_matmul(x, y)
