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

import torch
from torch.export import ExportedProgram

from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassBase
from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassResult  # NOQA

# A dummy decomp table for running ExportedProgram.run_decompositions without
# any op decompositions but just aot_export_module. Due to the check in
# run_decompositions, if None or an empty dict is passed as decomp_table,
# it will run the default aten-coreaten decompositions. Therefore a non-empty
# dummy decomp table is needed.
# Ref: https://github.com/pytorch/pytorch/blob/db895ace1d36726e64781774f53b3d3098206116/torch/export/exported_program.py#L543
_dummy_decomp_table = {
    torch._ops.OperatorBase(): lambda: None,
}


class CanonicalizePass(ExportedProgramPassBase):

  def call(self, exported_program: ExportedProgram):
    exported_program = exported_program.run_decompositions(_dummy_decomp_table)
    return ExportedProgramPassResult(exported_program, True)
