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
"""Provides lowering for coreaten to mlir stablehlo op: Convolution"""

import math
from typing import Optional

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch

from .registry import lower


# convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride,
#   SymInt[] padding, SymInt[] dilation, bool transposed,
#   SymInt[] output_padding, SymInt groups) -> Tensor
# @lower(torch.ops.aten.convolution)
def _aten_convolution(
    lctx,
    lhs: ir.Value,
    rhs: ir.Value,
    bias: Optional[ir.Value],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
):
  if transposed:
    raise NotImplementedError("Transposed convolution is not implemented.")

  if bias is not None:
    raise NotImplementedError("Bias on convolution is not implemented.")

  # Stablehlo allows start and end padding for each dimension while aten only
  # allows symmetric padding and so only has one number per dimension.
  def make_padding(padding):
    return tuple((p, p) for p in padding)

  def create_conv_dimension_numbers():
    num_spatial_dims = len(lhs.type.shape) - 2
    spatial_dimensions = []
    for i in range(0, num_spatial_dims):
      spatial_dimensions.append(i + 2)

    dimension_numbers = stablehlo.ConvDimensionNumbers.get(
        input_batch_dimension=0,
        input_feature_dimension=1,
        input_spatial_dimensions=spatial_dimensions,
        kernel_input_feature_dimension=1,
        kernel_output_feature_dimension=0,
        kernel_spatial_dimensions=spatial_dimensions,
        output_batch_dimension=0,
        output_feature_dimension=1,
        output_spatial_dimensions=spatial_dimensions,
    )
    return dimension_numbers

  def infer_output_shape():
    lhs_type: ir.RankedTensorType = lhs.type
    lhs_shape: list[int] = lhs_type.shape
    rhs_shape: list[int] = rhs.type.shape

    # Input layout is: (N)CHW and Kernel layout is: (O)IHW
    output_shape = [lhs_shape[0], rhs_shape[0]]
    num_spatial_dims = len(lhs.type.shape) - 2

    # looping over the spatial dims (skipping the first 2 dims which are
    # batch and features)
    for spatial_dim in range(0, num_spatial_dims):
      dim_size = lhs_shape[spatial_dim + 2]
      kernel_dim_size = rhs_shape[spatial_dim + 2]

      # for example, a dilation of 2 increases the dimension size by 2
      dim_size *= dilation[spatial_dim]

      # padding added to both sides
      dim_size += 2 * padding[spatial_dim]

      output_dim_size = math.ceil(
          (dim_size - kernel_dim_size + 1) / stride[spatial_dim]
      )

      output_shape.append(output_dim_size)

    return output_shape

  lhs_type: ir.RankedTensorType = lhs.type

  op = stablehlo.ConvolutionOp(
      result=ir.RankedTensorType.get(
          infer_output_shape(), lhs_type.element_type
      ),
      lhs=lhs,
      rhs=rhs,
      dimension_numbers=create_conv_dimension_numbers(),
      feature_group_count=groups,
      batch_group_count=1,
      window_strides=stride,
      padding=make_padding(padding),
      lhs_dilation=(1,) * len(stride),
      rhs_dilation=dilation,
  )

  return op.result
