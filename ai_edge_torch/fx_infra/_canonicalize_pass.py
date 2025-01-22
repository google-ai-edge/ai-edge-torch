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
from ai_edge_torch.fx_infra import _safe_run_decompositions
from ai_edge_torch.fx_infra import pass_base
import torch


class CanonicalizePass(pass_base.ExportedProgramPassBase):

  def call(self, exported_program: torch.export.ExportedProgram):
    exported_program = _safe_run_decompositions.safe_run_decompositions(
        exported_program, {}
    )

    return pass_base.ExportedProgramPassResult(exported_program, True)
