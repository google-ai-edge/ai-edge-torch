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

import abc
import collections
from typing import Sequence, Union

import torch
from torch.fx.passes.infra.pass_base import PassBase
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import pass_result_wrapper
import torch.utils._pytree as pytree

FxPassBase = PassBase
FxPassResult = PassResult
ExportedProgramPassResult = collections.namedtuple(
    "ExportedProgramPassResult", ["exported_program", "modified"]
)


class ExportedProgramPassBase(abc.ABC):

  def __call__(
      self, exported_program: torch.export.ExportedProgram
  ) -> ExportedProgramPassResult:
    self.requires(exported_program)
    res = self.call(exported_program)
    self.ensures(exported_program)
    return res

  @abc.abstractmethod
  def call(
      self, exported_program: torch.export.ExportedProgram
  ) -> ExportedProgramPassResult:
    pass

  def requires(self, exported_program: torch.export.ExportedProgram) -> None:
    pass

  def ensures(self, exported_program: torch.export.ExportedProgram) -> None:
    pass


# TODO(cnchan): make a PassManager class.
def run_passes(
    exported_program: torch.export.ExportedProgram,
    passes: Sequence[Union[ExportedProgramPassBase, FxPassBase]],
) -> torch.export.ExportedProgram:
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
        exported_program = torch.export.ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=exported_program.graph_signature,
            state_dict=exported_program.state_dict,
            range_constraints=exported_program.range_constraints,
            module_call_graph=exported_program.module_call_graph,
            example_inputs=exported_program.example_inputs,
            verifiers=exported_program.verifiers,
            constants=exported_program.constants,
        )
  return exported_program
