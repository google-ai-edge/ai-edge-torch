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
# Common utility functions for data loading etc.
from dataclasses import dataclass
from ai_edge_torch.odml_torch import lowerings
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch


# Use torch.library.custom_op to define a new custom operator.
@torch.library.custom_op("ai_edge_torch::bmm_4d", mutates_args=())
def bmm_4d(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
  if not (lhs.ndim == 4 and rhs.ndim == 4):
    raise ValueError("bmm_4d requires LHS and RHS have rank 4.")
  d0_can_bcast = (
      lhs.shape[0] == rhs.shape[0] or lhs.shape[0] == 1 or rhs.shape[0] == 1
  )
  d1_can_bcast = (
      lhs.shape[1] == rhs.shape[1] or lhs.shape[1] == 1 or rhs.shape[1] == 1
  )
  if not (d0_can_bcast and d1_can_bcast):
    raise ValueError("bmm_4d requires that dimensions 0 and 1 can broadcast.")

  if not lhs.shape[-1] == rhs.shape[-1]:
    raise ValueError("bmm_4d requires LHS and RHS have same last dimension.")

  return torch.einsum("abcd,abed->abce", lhs, rhs)


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@torch.library.custom_op("ai_edge_torch::bmm_4d", mutates_args=())
def _bmm_4d_fake(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
  return torch.einsum("abcd,abed->abce", lhs, rhs)


# Use explicit fake implementation for bmm_4d because dynamo cannot derive the
# output's symbolic shape from the impl above.
@torch.library.register_fake("ai_edge_torch::bmm_4d")
def bmm_4d_fake(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
  a, b, c, d = lhs.shape
  a, b, e, d = rhs.shape
  return torch.empty([a, b, c, e], dtype=lhs.dtype)


@lowerings.lower(torch.ops.ai_edge_torch.bmm_4d)
def _bmm_4d_lower(
    lctx,
    lhs: ir.Value,
    rhs: ir.Value,
):
  dot_dnums = stablehlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=[0, 1],
      rhs_batching_dimensions=[0, 1],
      lhs_contracting_dimensions=(3,),
      rhs_contracting_dimensions=(3,),
  )
  return stablehlo.dot_general(
      ir.RankedTensorType.get(
          (
              lhs.type.shape[0],
              lhs.type.shape[1],
              lhs.type.shape[2],
              rhs.type.shape[2],
          ),
          lhs.type.element_type,
      ),
      lhs,
      rhs,
      dot_dnums,
  )
