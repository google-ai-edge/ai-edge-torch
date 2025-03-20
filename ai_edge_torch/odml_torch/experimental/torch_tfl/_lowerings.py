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
