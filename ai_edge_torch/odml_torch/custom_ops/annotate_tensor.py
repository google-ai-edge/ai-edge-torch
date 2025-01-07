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
"""Custom op for annotating tensor with arbitary string."""

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
import torch
from .. import _torch_library
from .. import lowerings


_torch_library.ODML_TORCH_LIB.define(
    "annotate_tensor(Tensor x, str data) -> Tensor"
)


@torch.library.impl(
    _torch_library.ODML_TORCH_LIB,
    "annotate_tensor",
    "CompositeExplicitAutograd",
)
def _annotate_tensor_impl(x: torch.Tensor, _: str):
  return x


@torch.library.impl(_torch_library.ODML_TORCH_LIB, "annotate_tensor", "Meta")
def _annotate_tensor_meta(x: torch.Tensor, _: str):
  return torch.empty_like(x)


@lowerings.lower(torch.ops.odml_torch.annotate_tensor)
def _annotate_tensor_lowering(lctx, x: ir.Value, data: str):
  return stablehlo.CustomCallOp(
      result=[x.type],
      inputs=[x],
      call_target_name=ir.StringAttr.get("custom_call.annotate_tensor"),
      backend_config=ir.StringAttr.get(str(data)),
  ).results[0]
