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
"""Torch export decompositions to run before lowering."""

import functools
from ai_edge_torch import fx_infra
import torch


fx_infra.decomp.update_pre_lower_decomp(
    torch._decomp.get_decompositions([
        torch.ops.aten.upsample_nearest2d,
        torch.ops.aten._native_batch_norm_legit.no_stats,
        torch.ops.aten._native_batch_norm_legit_functional,
        torch.ops.aten._adaptive_avg_pool2d,
        torch.ops.aten._adaptive_avg_pool3d,
        torch.ops.aten.grid_sampler_2d,
        torch.ops.aten.native_group_norm,
        torch.ops.aten.native_dropout,
        torch.ops.aten.reflection_pad1d,
        torch.ops.aten.reflection_pad2d,
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.replication_pad1d,
        torch.ops.aten.replication_pad2d,
        torch.ops.aten.replication_pad3d,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.addmm,
    ])
)

fx_infra.decomp.remove_pre_lower_decomp(torch.ops.aten.roll)

# Torch's default einsum impl/decompositions is less efficient and
# optimized through converter than JAX's impl. Disable einsum
# decomposition to use JAX bridge for a more efficient lowering.
fx_infra.decomp.remove_pre_lower_decomp(torch.ops.aten.einsum.default)


# Override noop aten op decompositions for faster run_decompositions.
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.alias.default, lambda x: x
)
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.detach.default, lambda x: x
)

# Override _safe_softmax decompositions with regular softmax.
# _safe_softmax introduces additional check-select ops to guard extreme
# input values to softmax, which could make the converted model inefficient
# on-device.
if hasattr(torch.ops.aten, "_safe_softmax"):
  fx_infra.decomp.add_pre_convert_decomp(
      torch.ops.aten._safe_softmax.default,
      torch.softmax,
  )


# Decomp torch.scatter into one_hot, broadcasting, mul, and selects.
# This is a more GPU-friendly implementation than default
# lowering via stablehlo.scatter or tfl.scatter_nd.
@functools.partial(
    fx_infra.decomp.add_pre_convert_decomp, torch.ops.aten.scatter.src
)
def _scatter_impl(
    self: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
  if dim < 0:
    dim = self.dim() + dim

  # --- 1. Slice `src` to match the shape of `index` ---
  slicing_idx_for_src = tuple(slice(s) for s in index.shape)
  src_sliced = src[slicing_idx_for_src]

  # --- 2. Compute updates for the relevant slice using one_hot ---
  num_classes = self.shape[dim]
  one_hot_indices = torch.nn.functional.one_hot(index, num_classes)
  slice_updates_unaggregated = src_sliced.unsqueeze(-1) * one_hot_indices
  slice_updates_summed = slice_updates_unaggregated.sum(dim=dim)
  slice_condition_summed = one_hot_indices.any(dim=dim)

  # --- 3. Permute the computed slice to the correct dimension order ---
  n_dims = self.dim()
  slice_updates = slice_updates_summed
  slice_condition = slice_condition_summed
  if n_dims > 1:
    permute_order = list(range(n_dims - 1))
    permute_order.insert(dim, n_dims - 1)
    slice_updates = slice_updates_summed.permute(permute_order)
    slice_condition = slice_condition_summed.permute(permute_order)

  # Pad the smaller tensors to match the shape of `self`.
  require_padding = True
  try:
    shape = torch.broadcast_shapes(slice_updates.shape, self.shape)
    if shape == self.shape:
      require_padding = False
  except RuntimeError:
    # Shapes are not broadcastable.
    require_padding = True
    pass

  if require_padding:
    pad_amounts = []
    for i in range(n_dims - 1, -1, -1):
      padding_needed = self.shape[i] - slice_updates.shape[i]
      # Add 0 for the "start" and the needed amount for the "end"
      pad_amounts.extend([0, padding_needed])
    updates_tensor = torch.nn.functional.pad(
        slice_updates, pad_amounts, "constant", 0
    )
    condition_mask = torch.nn.functional.pad(
        slice_condition, pad_amounts, "constant", 0
    )
  else:
    updates_tensor = slice_updates
    condition_mask = slice_condition

  # --- 5. Use `torch.where` on correctly-sized tensors ---
  # IMPORTANT NOTE: When indices are not unique, the behavior of torch scatter
  # is non-deterministic (one of the values from src will be picked
  # arbitrarily)
  result = torch.where(condition_mask, updates_tensor, self)
  return result
