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
# Dynamic update slice op for KV cache update.
from dataclasses import dataclass
from typing import Sequence
from ai_edge_torch.odml_torch import lowerings
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch


# Use torch.library.custom_op to define a new custom operator.
# TODO: Update impl for multiple non-trivial start_indices
@torch.library.custom_op("ai_edge_torch::dynamic_update_slice", mutates_args=())
def dynamic_update_slice(
    in_tensor: torch.Tensor,
    update: torch.Tensor,
    start_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
  compare_size = torch.tensor(in_tensor.size()) == torch.tensor(update.size())
  mismatch = torch.nonzero(~compare_size, as_tuple=False)
  dim = mismatch[0].item() if len(mismatch) > 0 else 0
  start = start_indices[dim].item()
  end = start + update.shape[dim]
  indices = torch.arange(start, end).to(torch.long)
  return in_tensor.index_copy(dim, indices, update)


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@torch.library.custom_op("ai_edge_torch::dynamic_update_slice", mutates_args=())
def _(
    in_tensor: torch.Tensor,
    update: torch.Tensor,
    start_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
  return in_tensor.clone().detach()


# Use explicit fake implementation for dynamic_update_slice because dynamo
# cannot derive the output's symbolic shape from the impl above.
@torch.library.register_fake("ai_edge_torch::dynamic_update_slice")
def dynamic_update_slice_fake(
    in_tensor: torch.Tensor,
    update: torch.Tensor,
    start_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
  return torch.empty(in_tensor.size(), dtype=in_tensor.dtype)


@lowerings.lower(torch.ops.ai_edge_torch.dynamic_update_slice)
def _dynamic_update_slice_lower(
    lctx,
    in_tensor: ir.Value,
    update: ir.Value,
    start_indices: Sequence[ir.Value],
):
  return stablehlo.dynamic_update_slice(in_tensor, update, start_indices)
