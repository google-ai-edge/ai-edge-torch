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

from ai_edge_torch.fx_infra import _canonicalize_pass
from ai_edge_torch.fx_infra import _safe_run_decompositions
from ai_edge_torch.fx_infra import decomp
from ai_edge_torch.fx_infra import graph_utils
from ai_edge_torch.fx_infra import pass_base


PassBase = pass_base.PassBase
PassResult = pass_base.PassResult
FxPassBase = pass_base.FxPassBase
FxPassResult = pass_base.FxPassResult
ExportedProgramPassBase = pass_base.ExportedProgramPassBase
ExportedProgramPassResult = pass_base.ExportedProgramPassResult
run_passes = pass_base.run_passes

CanonicalizePass = _canonicalize_pass.CanonicalizePass
safe_run_decompositions = _safe_run_decompositions.safe_run_decompositions
