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
from ai_edge_torch import odml_torch
from ai_edge_torch.odml_torch import optimization_barrier as optimization_barrier_lib  # Import to register the op.
import torch

from absl.testing import absltest as googletest

optimization_barrier = optimization_barrier_lib.optimization_barrier


class TestOptimizationBarrier(googletest.TestCase):
  """Test optimization barrier op implementation and lowering."""

  def test_optimization_barrier_op(self):

    class TestModel(torch.nn.Module):

      def forward(self, x, y):
        x, _ = optimization_barrier(x, y)
        return x

    x = torch.randn(1, 5)
    ep = torch.export.export(TestModel().eval(), (x, x))
    mlir = odml_torch.export.exported_program_to_mlir(ep)
    mlir_text = mlir.get_text()
    self.assertEqual(
        mlir_text.count(
            "stablehlo.optimization_barrier %arg1, %arg1 : tensor<1x5xf32>,"
            " tensor<1x5xf32>"
        ),
        1,
    )


if __name__ == "__main__":
  googletest.main()
