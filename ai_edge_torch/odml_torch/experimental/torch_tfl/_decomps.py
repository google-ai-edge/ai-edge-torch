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


@register_decomp(torch.ops.aten.add.Tensor)
def _aten_add_tensor_decomp(x, y, alpha=1):
  out = torch.ops.tfl.add(x, y)

  if alpha != 1:
    alpha = torch.scalar_tensor(alpha, dtype=out.dtype)
    out = torch.ops.tfl.add(x, torch.ops.tfl.mul(y, alpha))

  return out


@register_decomp(torch.ops.aten.sub.Tensor)
def _aten_sub_tensor_decomp(x, y, alpha=1):
  out = torch.ops.tfl.sub(x, y)

  if alpha != 1:
    alpha = torch.scalar_tensor(alpha, dtype=out.dtype)
    out = torch.ops.tfl.sub(x, torch.ops.tfl.mul(y, alpha))

  return out


@register_decomp(torch.ops.aten.mul.Tensor)
def _aten_mul_tensor_decomp(x, y):
  return torch.ops.tfl.mul(x, y)


@register_decomp(torch.ops.aten.div.Tensor)
def _aten_div_tensor_decomp(x, y):
  return torch.ops.tfl.div(x, y)


@register_decomp(torch.ops.aten.gt.Tensor)
def _aten_gt_tensor_decomp(x, y):
  return torch.ops.tfl.greater(x, y)


@register_decomp(torch.ops.aten.lt.Tensor)
def _aten_lt_tensor_decomp(x, y):
  return torch.ops.tfl.less(x, y)


@register_decomp(torch.ops.aten.maximum.default)
def _aten_maximum_tensor_decomp(x, y):
  return torch.ops.tfl.maximum(x, y)


@register_decomp(torch.ops.aten.minimum.default)
def _aten_minimum_tensor_decomp(x, y):
  return torch.ops.tfl.minimum(x, y)


@register_decomp(torch.ops.aten.sin.default)
def _aten_sin_decomp(x):
  return torch.ops.tfl.sin(x)


@register_decomp(torch.ops.aten.cos.default)
def _aten_cos_decomp(x):
  return torch.ops.tfl.cos(x)


@register_decomp(torch.ops.aten.rsqrt.default)
def _aten_rsqrt_decomp(x):
  return torch.ops.tfl.rsqrt(x)


@register_decomp(torch.ops.aten.gelu.default)
def _aten_gelu_decomp(x, approximate="none"):
  return torch.ops.tfl.gelu(x, approximate != "none")
