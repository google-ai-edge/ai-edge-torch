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

import unittest

import torch
import torch_xla

from ai_edge_torch.hlfb import mark_pattern


def _export_stablehlo_mlir(model, args=None):
  if not isinstance(model, torch.export.ExportedProgram):
    ep = torch.export.export(model, args)
  else:
    ep = model
  stablehlo_gm = torch_xla.stablehlo.exported_program_to_stablehlo(ep)
  return stablehlo_gm.get_stablehlo_text()


class TestMarkPattern(unittest.TestCase):

  def test_mark_pattern(self):

    class TestModel(torch.nn.Module):

      def forward(self, x):
        return x * x + x + x

    pattern = mark_pattern.Pattern(
        "test.add",
        lambda a, b: a + b,
        export_args=(torch.rand(2, 2), torch.rand(2, 2)),
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20),)
    exported_program = torch.export.export(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _export_stablehlo_mlir(exported_program)

    self.assertEqual(mlir.count('stablehlo.composite "test.add"'), 2)

  def test_mark_pattern_with_attr_builder(self):
    class TestModel(torch.nn.Module):

      def forward(self, x):
        return x * x * x + x - x * x + x

    pattern = mark_pattern.Pattern(
        "test.add",
        lambda a, b: a + b,
        export_args=(torch.rand(2, 2), torch.rand(2, 2)),
        attr_builder=lambda *args: {"alias": "test.test_add"},
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20),)
    exported_program = torch.export.export(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _export_stablehlo_mlir(exported_program)

    self.assertEqual(mlir.count('stablehlo.composite "test.add"'), 2)
    self.assertEqual(mlir.count('composite_attributes = {alias = "test.test_add"}'), 2)

  def test_mark_pattern_with_scalar_attr_tracker(self):
    class TestModel(torch.nn.Module):

      def forward(self, x):
        r = x
        for idx in range(5):
          r = torch.nn.LogSoftmax(dim=idx % 2)(r) * x
        return r

    pattern = mark_pattern.Pattern(
        "test.log_softmax",
        lambda x, dim: torch.nn.functional.log_softmax(x, dim=dim),
        export_args=(torch.rand(10, 10, 10), 1),
        scalar_attr_trackers=[
            mark_pattern.ScalarAttrTracker("dim", pattern_arg_pos=1)
            .track(0)
            .track(1)
            .track(2),
        ],
    )

    model = TestModel().eval()
    args = (torch.rand(10, 10),)
    exported_program = torch.export.export(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _export_stablehlo_mlir(exported_program)

    self.assertEqual(mlir.count('stablehlo.composite "test.log_softmax"'), 5)
    self.assertEqual(mlir.count("composite_attributes = {dim = 0 : i64}"), 3)
    self.assertEqual(mlir.count("composite_attributes = {dim = 1 : i64}"), 2)

  def test_mark_tangent_model_and_pattern_input(self):
    class TestModel(torch.nn.Module):

      def forward(self, x, y):
        z = torch.ops.aten.relu(x)
        z = z + y
        return z

    pattern = mark_pattern.Pattern(
        "test.relu",
        lambda x: torch.ops.aten.relu(x),
        export_args=(torch.rand(2, 2),),
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20), torch.rand(20, 20))
    exported_program = torch.export.export(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _export_stablehlo_mlir(exported_program)

    self.assertEqual(mlir.count('stablehlo.composite "test.relu'), 1)


if __name__ == "__main__":
  unittest.main()
