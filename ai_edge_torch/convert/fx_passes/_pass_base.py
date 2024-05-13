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
from collections import namedtuple

import torch
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassBase as FxPassBase
from torch.fx.passes.infra.pass_base import PassResult as FxPassResult


class ExportedProgramPassResult(
    namedtuple("ExportedProgramPassResult", ["exported_program", "modified"])
):

  def __new__(cls, exported_program, modified):
    return super().__new__(cls, exported_program, modified)


class ExportedProgramPassBase(abc.ABC):

  def __call__(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
    self.requires(exported_program)
    res = self.call(exported_program)
    self.ensures(exported_program)
    return res

  @abc.abstractmethod
  def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
    pass

  def requires(self, exported_program: ExportedProgram) -> None:
    pass

  def ensures(self, exported_program: ExportedProgram) -> None:
    pass
