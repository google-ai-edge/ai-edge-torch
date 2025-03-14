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
"""Torch-TFL op definitions and fake implementations."""
import re
from typing import Sequence
from ai_edge_torch.odml_torch.experimental.torch_tfl import torch_library_utils
import torch

custom_op_with_fake = torch_library_utils.custom_op_with_fake


@custom_op_with_fake("tfl::batch_matmul")
def tfl_batch_matmul(
    x: torch.Tensor, y: torch.Tensor, adj_x: bool = False, adj_y: bool = False
) -> torch.Tensor:
  if x.ndim < 2 or y.ndim < 2:
    raise ValueError("Input tensors must have at least 2 dimensions.")
  if adj_x:
    x = torch.transpose(x, -1, -2)
  if adj_y:
    y = torch.transpose(y, -1, -2)
  return torch.matmul(x, y)


@custom_op_with_fake("tfl::slice")
def tfl_slice(
    input: torch.Tensor, begin: Sequence[int], size: Sequence[int]
) -> torch.Tensor:
  assert len(begin) == len(size) == input.ndim

  slices = [slice(i, i + l) for i, l in zip(begin, size)]
  return input[tuple(slices)].clone()


@torch.library.custom_op("tfl::slice.tensor", mutates_args=())
def tfl_slice_tensor(
    input: torch.Tensor,
    begin: torch.Tensor,
    size: torch.Tensor,
    *,
    shape: str = "",
) -> torch.Tensor:
  assert begin.ndim == size.ndim == 1
  assert begin.numel() == size.numel() == input.ndim
  assert begin.dtype == torch.int32 and size.dtype == torch.int32
  assert not shape or shape.count(",") == input.ndim - 1

  slices = [slice(i, i + l) for i, l in zip(begin.tolist(), size.tolist())]
  return input[tuple(slices)].clone()


@torch.library.register_fake("tfl::slice.tensor")
def tfl_slice_tensor_fake(
    input: torch.Tensor,
    begin: torch.Tensor,
    size: torch.Tensor,
    *,
    shape: str = "",
) -> torch.Tensor:
  ctx = torch.library.get_ctx()
  shape_str = shape
  if not shape_str:
    shape_str = ",".join(["?" for _ in range(input.ndim)])

  shape = []
  shape_symbols = shape_str.split(",")
  for sym in shape_symbols:
    if re.match(r"\d+", sym):
      shape.append(int(sym))
    else:
      nnz = ctx.new_dynamic_size()
      shape.append(nnz)
  return input.new_empty(shape, dtype=input.dtype)
