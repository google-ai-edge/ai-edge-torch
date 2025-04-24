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
