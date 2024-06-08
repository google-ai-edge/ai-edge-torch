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

from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_manager import pass_result_wrapper
import torch.utils._pytree as pytree

from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassBase
from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassResult  # NOQA
from ai_edge_torch.convert.fx_passes._pass_base import FxPassBase
from ai_edge_torch.convert.fx_passes._pass_base import FxPassResult
from ai_edge_torch.convert.fx_passes.build_aten_composite_pass import BuildAtenCompositePass  # NOQA
from ai_edge_torch.convert.fx_passes.build_interpolate_composite_pass import BuildInterpolateCompositePass  # NOQA
from ai_edge_torch.convert.fx_passes.canonicalize_pass import CanonicalizePass
from ai_edge_torch.convert.fx_passes.inject_mlir_debuginfo_pass import InjectMlirDebuginfoPass  # NOQA
from ai_edge_torch.convert.fx_passes.optimize_layout_transposes_pass import OptimizeLayoutTransposesPass  # NOQA


# TODO(cnchan): make a PassManager class.
def run_passes(
    exported_program: ExportedProgram,
    passes: Sequence[Union[ExportedProgramPassBase, FxPassBase]],
) -> ExportedProgram:
  passes, _ = pytree.tree_flatten(passes)
  for pass_ in passes:
    if not isinstance(pass_, ExportedProgramPassBase):
      pass_ = pass_result_wrapper(pass_)
    if isinstance(pass_, ExportedProgramPassBase):
      exported_program = pass_(exported_program).exported_program
    else:
      gm = exported_program.graph_module
      gm, modified = pass_(gm)
      if modified and gm is not exported_program.graph_module:
        exported_program = ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=exported_program.graph_signature,
            state_dict=exported_program.state_dict,
            range_constraints=exported_program.range_constraints,
            module_call_graph=exported_program.module_call_graph,
            example_inputs=exported_program.example_inputs,
            verifier=exported_program.verifier,
            constants=exported_program.constants,
        )
  return exported_program
