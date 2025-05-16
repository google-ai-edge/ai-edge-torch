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
"""Tests for RemoveNonUserOutputsPass."""

from typing import Callable, Union

from ai_edge_torch import fx_infra
from ai_edge_torch._convert import fx_passes
import torch

from absl.testing import absltest as googletest


def export_with_pass(
    func: Union[torch.nn.Module, Callable[..., torch.Tensor]],
    export_args: list[torch.Tensor],
) -> torch.export.ExportedProgram:
  """Exports a function with OptimizeLayoutTransposesPass."""
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = fx_infra.safe_run_decompositions(
      exported_program,
      fx_infra.decomp.pre_convert_decomp(),
  )
  exported_program = fx_infra.run_passes(
      exported_program,
      [fx_passes.EliminateDeadCodePass()],
  )
  return exported_program


class TestEliminateDeadCodePass(googletest.TestCase):
  """Tests for EliminateDeadCodePass."""

  def test_eliminate_unused_rand_nodes(self):

    def f(x):
      a = torch.ops.aten.rand.default((10, 10))
      b = torch.ops.aten.rand.default((10, 10))
      _ = a + b
      x = x + 1
      return x

    exported_program = export_with_pass(f, (torch.rand(10, 10),))

    ops = [node.target for node in exported_program.graph.nodes]
    self.assertNotIn(torch.ops.aten.rand.default, ops)


if __name__ == "__main__":
  googletest.main()
