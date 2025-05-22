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
from typing import Any, Sequence

from ai_edge_torch.odml_torch.experimental.torch_tfl import torch_library_utils
import numpy as np
import torch
from torch.fx.experimental.symbolic_shapes import has_free_symbols, is_symbolic


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


@custom_op_with_fake("tfl::add", schema="(Tensor x, Any y) -> Tensor")
def tfl_add(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.add(x, y)


@custom_op_with_fake("tfl::sub", schema="(Tensor x, Any y) -> Tensor")
def tfl_sub(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.sub(x, y)


@custom_op_with_fake("tfl::mul", schema="(Tensor x, Any y) -> Tensor")
def tfl_mul(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.mul(x, y)


@custom_op_with_fake("tfl::div", schema="(Tensor x, Any y) -> Tensor")
def tfl_div(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.div(x, y)


@custom_op_with_fake("tfl::pow", schema="(Any x, Any y) -> Tensor")
def tfl_pow(x: Any, y: Any) -> torch.Tensor:
  return torch.pow(x, y)


@custom_op_with_fake("tfl::logical_and")
def tfl_logical_and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.logical_and(x, y)


@custom_op_with_fake("tfl::greater")
def tfl_greater(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.gt(x, y)


@custom_op_with_fake("tfl::less")
def tfl_less(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.lt(x, y)


@custom_op_with_fake("tfl::maximum")
def tfl_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.maximum(x, y)


@custom_op_with_fake("tfl::minimum")
def tfl_minimum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.minimum(x, y)


@custom_op_with_fake("tfl::sin")
def tfl_sin(x: torch.Tensor) -> torch.Tensor:
  return torch.sin(x)


@custom_op_with_fake("tfl::cos")
def tfl_cos(x: torch.Tensor) -> torch.Tensor:
  return torch.cos(x)


@custom_op_with_fake("tfl::rsqrt")
def tfl_rsqrt(x: torch.Tensor) -> torch.Tensor:
  return torch.rsqrt(x)


@custom_op_with_fake("tfl::gelu")
def tfl_gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
  gelu_approximate = "tanh" if approximate else "none"
  return torch.nn.functional.gelu(x, approximate=gelu_approximate)


@custom_op_with_fake("tfl::transpose")
def tfl_transpose(input: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
  assert len(perm) == input.ndim

  return torch.permute(input, perm).clone()


def _normalize_shape(
    tensor_input: torch.Tensor, shape: Sequence[int]
) -> Sequence[int]:
  """Normalize the size for the -1 dimension in the "shape".

  Args:
      tensor_input: The input tensor.
      shape: The desired shape, which may contain a -1 to indicate an inferred
        dimension.

  Returns:
      The inferred shape.

  Raises:
      ValueError: If the shape is invalid or cannot be inferred.
  """
  inferred_shape = list(shape)
  if -1 in inferred_shape:
    numel = tensor_input.numel()
    product = 1
    neg_one_idx = -1
    for i, dim in enumerate(inferred_shape):
      if dim == -1:
        if neg_one_idx != -1:
          raise ValueError("Only one dimension can be inferred (-1)")
        neg_one_idx = i
      elif dim >= 0:
        product *= dim
      else:
        raise ValueError(
            "Shape dimensions must be non-negative or -1 for inference"
        )

    if neg_one_idx != -1:
      if product == 0:
        if numel != 0:
          raise ValueError(
              "Cannot infer dimension for non-zero input size when other"
              " dimensions multiply to zero"
          )
        inferred_shape[neg_one_idx] = 0
      else:
        if numel % product != 0:
          raise ValueError(
              f"Input size {numel} not divisible by product of known dimensions"
              f" {product}"
          )
        inferred_shape[neg_one_idx] = numel // product

  # Ensure the inferred shape still matches the total number of elements
  if np.prod(inferred_shape) != tensor_input.numel():
    raise ValueError(
        f"Calculated shape {inferred_shape} does not match input numel"
        f" {tensor_input.numel()}"
    )

  return inferred_shape


@torch.library.custom_op("tfl::reshape", mutates_args=())
def tfl_reshape(input: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
  inferred_shape = _normalize_shape(input, shape)
  return input.view(inferred_shape).clone()


# Use explicit fake implementation for tfl.reshape because dynamo cannot
# derive the output's symbolic shape from the impl above.
@torch.library.register_fake("tfl::reshape")
def tfl_reshape_fake(input: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
  inferred_shape = _normalize_shape(input, shape)
  return torch.empty(inferred_shape, dtype=input.dtype)


@torch.library.custom_op(
    "tfl::range",
    mutates_args=(),
    schema="(Any start, Any limit, Any delta) -> Tensor",
)
def tfl_range(start: Any, limit: Any, delta: Any) -> torch.Tensor:
  return torch.arange(start, limit, delta)


# Use explicit fake implementation for tfl.range because dynamo cannot
# derive the output's symbolic shape from the impl above.
@torch.library.register_fake("tfl::range")
def tfl_range_fake(start: Any, limit: Any, delta: Any) -> torch.Tensor:
  # Determine output dtype
  dt = torch.int64
  for val_check in [start, limit, delta]:
    if isinstance(val_check, float):  # Python float
      dt = torch.get_default_dtype()
      break
    if isinstance(val_check, torch.Tensor):  # PyTorch tensor
      if val_check.is_floating_point():
        dt = torch.get_default_dtype()
        break
    elif is_symbolic(val_check):  # Symbolic number (SymInt or SymFloat)
      temp_tensor_for_dtype_check = torch.tensor(val_check)
      if temp_tensor_for_dtype_check.is_floating_point():
        dt = torch.get_default_dtype()
        break

  s, l, d = start, limit, delta
  numerator = torch.sub(l, s)

  # Determine if delta is a concrete zero
  is_concrete_zero_delta = False
  if isinstance(d, (int, float)) and d == 0:
    is_concrete_zero_delta = True
  elif (
      isinstance(d, torch.Tensor) and not has_free_symbols(d) and d.numel() == 1
  ):
    # d is a scalar tensor with a concrete shape. Check its value.
    delta_item = d.item()
    # Only if delta_item is NOT symbolic AND is zero, then treat as concrete zero.
    if not is_symbolic(delta_item) and delta_item == 0:
      is_concrete_zero_delta = True

  if is_concrete_zero_delta:
    final_size_float = torch.tensor(0.0, dtype=torch.get_default_dtype())
  else:
    # General case for non-zero or symbolic delta
    num_steps_float = torch.true_divide(numerator, d)
    size_float = torch.ceil(num_steps_float)
    # The zero tensor for maximum should match the type of size_float (usually float)
    # or the default float type if size_float is not a tensor (e.g. SymFloat)
    dtype_for_zero = (
        size_float.dtype
        if isinstance(size_float, torch.Tensor)
        else torch.get_default_dtype()
    )
    zero_for_max = torch.tensor(0.0, dtype=dtype_for_zero)
    final_size_float = torch.maximum(size_float, zero_for_max)

  # Convert to int/SymInt for shape
  is_final_size_symbolic_scalar = False
  # Check if final_size_float itself is a symbolic scalar (e.g., SymFloat)
  if is_symbolic(final_size_float):
    is_final_size_symbolic_scalar = True
  # Else, check if it's a 0-dim tensor containing a symbolic scalar
  elif (
      isinstance(final_size_float, torch.Tensor)
      and final_size_float.numel() == 1
      and has_free_symbols(final_size_float)  # Check the tensor directly
  ):
    is_final_size_symbolic_scalar = True

  if is_final_size_symbolic_scalar:
    # Extract the symbolic value if it's a tensor, then convert to SymInt
    value_to_convert = (
        final_size_float.item()
        if isinstance(final_size_float, torch.Tensor)
        else final_size_float
    )
    output_size = torch.sym_int(value_to_convert)
  else:
    # final_size_float is concrete (either a Python float/int or a concrete tensor)
    # Ensure it's a tensor for consistent handling via .item() and .all()
    if isinstance(final_size_float, torch.Tensor):
      final_size_tensor = final_size_float
    else:
      # This path is less likely given torch ops, but for robustness:
      final_size_tensor = torch.tensor(
          final_size_float, dtype=torch.get_default_dtype()
      )

    # Now final_size_tensor is a concrete tensor.
    if not torch.isfinite(final_size_tensor).all():  # This is now safe.
      output_size = 0
    else:
      # Ensure item is converted to int after checking for finiteness
      output_size = int(final_size_tensor.item())

  final_shape = (output_size,)
  return torch.empty(final_shape, dtype=dt)


@custom_op_with_fake("tfl::softmax")
def tfl_softmax(x: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.softmax(x, dim=-1)


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
