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

import torch
import torch.utils._pytree as pytree
import torchvision

from ai_edge_torch.convert.fx_passes import CanonicalizePass
from ai_edge_torch.convert.fx_passes import OptimizeLayoutTransposesPass
from ai_edge_torch.convert.fx_passes import run_passes


def export_with_pass(
    func: Union[torch.nn.Module, Callable], export_args
) -> torch.export.ExportedProgram:
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
          OptimizeLayoutTransposesPass(),
          CanonicalizePass(),
      ],
  )
  return exported_program


class TestOptimizeLayoutTransposesPass(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)

  def assert_outputs_allclose(self, m1, m2, args):
    out1 = m1(*args)
    out2 = m2(*args)
    out1, _ = pytree.tree_flatten(out1)
    out2, _ = pytree.tree_flatten(out2)
    self.assertEqual(len(out1), len(out2))
    for o1, o2 in zip(out1, out2):
      self.assertTrue(torch.allclose(o1, o2, atol=1e-5))

  def assert_nodes_ops_equal(
      self,
      exported_program: torch.export.ExportedProgram,
      ops: list[Callable],
  ):
    nodes = [
        node.target
        for node in exported_program.graph.nodes
        if node.op == 'call_function'
    ]
    self.assertEqual(nodes, ops)

  def test_torchvision_mobilenet_v3_small(self):
    model = torchvision.models.mobilenet_v3_small().eval()
    forward_args = lambda: (torch.rand(1, 3, 224, 224),)

    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(model, exported_program.module(), forward_args())

  def test_torchvision_resnet18(self):
    model = torchvision.models.resnet18().eval()
    forward_args = lambda: (torch.rand(1, 3, 224, 224),)

    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(model, exported_program.module(), forward_args())

  # TODO(cnchan): Add more tests.


if __name__ == '__main__':
  unittest.main()
