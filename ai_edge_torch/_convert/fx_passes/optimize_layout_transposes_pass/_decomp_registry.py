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
"""Remove decompositions for ops to keep in layout optimization."""
from ai_edge_torch import fx_infra
import torch

__all__ = []

aten = torch.ops.aten

_OPS_TO_KEEP = [
    aten.conv2d,
    aten.max_pool2d,
    aten._softmax.default,
    aten.group_norm.default,
    aten.native_group_norm.default,
    aten.reflection_pad2d.default,
]

for op in _OPS_TO_KEEP:
  fx_infra.decomp.remove_pre_convert_decomp(op)
