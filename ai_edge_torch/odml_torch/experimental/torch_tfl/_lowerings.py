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
"""Torch-TFL op to MLIR lowerings."""

from typing import Sequence

from ai_edge_torch import odml_torch
from ai_edge_torch.odml_torch.experimental.torch_tfl import _ops
from ai_edge_torch.odml_torch.lowerings import registry
from ai_edge_torch.odml_torch.lowerings import utils as lowering_utils
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import numpy as np
import torch


lower = registry.lower
LoweringContext = odml_torch.lowerings.context.LoweringContext


def _ir_operation(
    name: str,
    results: Sequence[ir.Type],
    operands: Sequence[ir.Value] | None = None,
    attributes: dict[str, ir.Attribute] | None = None,
):
  """Helper function to create an IR operation in StableHLO CustomCall carrier."""
  if not operands:
    operands = []
  attributes = ir.DictAttr.get(attributes if attributes else {})
  return stablehlo.custom_call(
      result=results,
      inputs=operands,
      call_target_name=ir.StringAttr.get(name),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(str(attributes)),
  )


@lower(torch.ops.tfl.batch_matmul.default)
def _tfl_batch_matmul_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    y: ir.Value,
    adj_x: bool = False,
    adj_y: bool = False,
) -> ir.Value:
  return _ir_operation(
      "tfl.batch_matmul",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, y],
      attributes={
          "adj_x": ir.BoolAttr.get(adj_x),
          "adj_y": ir.BoolAttr.get(adj_y),
          "asymmetric_quantize_inputs": ir.BoolAttr.get(False),
      },
  )


@lower(torch.ops.tfl.add.default)
def _tfl_add_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  return _ir_operation(
      "tfl.add",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.sub.default)
def _tfl_sub_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  return _ir_operation(
      "tfl.sub",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.mul.default)
def _tfl_mul_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  return _ir_operation(
      "tfl.mul",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.div.default)
def _tfl_div_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  return _ir_operation(
      "tfl.div",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.greater.default)
def _tfl_greater_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.greater",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.less.default)
def _tfl_less_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.less",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.maximum.default)
def _tfl_maximum_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.maximum",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.minimum.default)
def _tfl_minimum_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.minimum",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.sin.default)
def _tfl_sin_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.sin",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.cos.default)
def _tfl_cos_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.cos",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.rsqrt.default)
def _tfl_rsqrt_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.rsqrt",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.gelu.default)
def _tfl_gelu_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    approximate: bool = False,
) -> ir.Value:
  return _ir_operation(
      "tfl.gelu",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
      attributes={
          "approximate": ir.BoolAttr.get(approximate),
      },
  )


@lower(torch.ops.tfl.transpose.default)
def _tfl_transpose_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    perm: Sequence[int],
) -> ir.Value:
  constant_perm = lowering_utils.numpy_array_constant(
      np.array(perm, dtype=np.int32)
  )
  return _ir_operation(
      "tfl.transpose",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, constant_perm],
  )


@lower(torch.ops.tfl.softmax.default)
def _tfl_softmax_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    beta: float = 1.0,
) -> ir.Value:
  return _ir_operation(
      "tfl.softmax",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
      attributes={
          "beta": ir.FloatAttr.get(ir.F32Type.get(), beta),
      },
  )
