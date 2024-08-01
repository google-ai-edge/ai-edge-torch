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

from typing import Callable, Union
import unittest

from ai_edge_torch.convert.fx_passes import CanonicalizePass
from ai_edge_torch.convert.fx_passes import InjectMlirDebuginfoPass
from ai_edge_torch.convert.fx_passes import run_passes
import torch
import torch_xla


def _export_to_stablehlo_with_composite(
    func: Union[torch.nn.Module, Callable], export_args
):
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = run_passes(
      exported_program,
      [
          InjectMlirDebuginfoPass(),
          CanonicalizePass(),
      ],
  )

  return torch_xla.stablehlo.exported_program_to_stablehlo(
      exported_program
  ).get_stablehlo_text()


class TestInjectMlirDebuginfoPass(unittest.TestCase):

  def test_write_torch_layers_debuginfo(self):
    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax()

      def forward(self, x, y):
        z = x + y
        z = self.softmax(z)
        return z

    stablehlo = _export_to_stablehlo_with_composite(
        SampleModel().eval(), (torch.rand(10, 10), torch.rand(10, 10))
    )
    self.assertTrue(
        'SampleModel/torch.nn.modules.activation.Softmax_softmax;"' in stablehlo
    )
    self.assertTrue('SampleModel;"' in stablehlo)


if __name__ == '__main__':
  unittest.main()
