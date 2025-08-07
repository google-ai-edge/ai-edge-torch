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

from typing import Sequence, Union

from ai_edge_torch._convert.fx_passes.build_aten_composite_pass import BuildAtenCompositePass
from ai_edge_torch._convert.fx_passes.cast_inputs_bf16_to_f32_pass import CastInputsBf16ToF32Pass
from ai_edge_torch._convert.fx_passes.cast_inputs_i64_to_i32_pass import CastInputsI64ToI32Pass
from ai_edge_torch._convert.fx_passes.eliminate_dead_code_pass import EliminateDeadCodePass
from ai_edge_torch._convert.fx_passes.inject_mlir_debuginfo_pass import InjectMlirDebuginfoPass
from ai_edge_torch._convert.fx_passes.optimize_layout_transposes_pass import OptimizeLayoutTransposesPass
from ai_edge_torch._convert.fx_passes.remove_non_user_outputs_pass import RemoveNonUserOutputsPass
from ai_edge_torch.fx_infra import CanonicalizePass
