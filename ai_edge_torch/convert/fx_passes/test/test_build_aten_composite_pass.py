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
import torch_xla

from ai_edge_torch.convert.fx_passes import BuildAtenCompositePass
from ai_edge_torch.convert.fx_passes import CanonicalizePass
from ai_edge_torch.convert.fx_passes import run_passes


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
          BuildAtenCompositePass(),
          CanonicalizePass(),
      ],
  )

  return torch_xla.stablehlo.exported_program_to_stablehlo(
      exported_program
  ).get_stablehlo_text()


class TestBuildAtenCompositePass(unittest.TestCase):

  def test_hardswish_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.Hardswish()(x), (torch.rand(10, 10),)
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "aten.hardswish.default"'), 1)

  def test_hardswish_op(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.ops.aten.hardswish.default(x), (torch.rand(10, 10),)
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "aten.hardswish.default"'), 1)

  def test_avg_pool2d_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.AvgPool2d(
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[0, 0],
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=None,
        )(x),
        (torch.rand(1, 3, 6, 6),),
    )
    self.assertEqual(
        stablehlo.count('stablehlo.composite "aten.avg_pool2d.default"'), 1
    )

  def test_avg_pool2d_op(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.avg_pool2d(
            x,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=None,
        ),
        (torch.rand(1, 3, 6, 6),),
    )
    self.assertEqual(
        stablehlo.count('stablehlo.composite "aten.avg_pool2d.default"'), 1
    )

  def test_avg_pool2d_ceil_mode(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.functional.avg_pool2d(
            x,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            ceil_mode=True,
            count_include_pad=True,
            divisor_override=None,
        ),
        (torch.rand(1, 3, 6, 6),),
    )
    self.assertEqual(
        stablehlo.count('stablehlo.composite "aten.avg_pool2d.default"'), 1
    )

  def test_gelu_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.GELU()(x), (torch.rand(10, 10),)
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "aten.gelu.default"'), 1)

  def test_approximate_gelu_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda x: torch.nn.GELU('tanh')(x), (torch.rand(10, 10),)
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "aten.gelu.default"'), 1)

  def test_embedding_lookup_layer(self):
    stablehlo = _export_to_stablehlo_with_composite(
        torch.nn.Embedding(10, 10), (torch.full((1, 10), 0, dtype=torch.long),)
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "odml.embedding_lookup"'), 1)

  def test_embedding_lookup_op(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda *x: torch.ops.aten.embedding.default(*x),
        (torch.rand(10, 10), torch.full((1, 10), 0, dtype=torch.long)),
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "odml.embedding_lookup"'), 1)

  def test_embedding_lookup_functional(self):
    stablehlo = _export_to_stablehlo_with_composite(
        lambda *x: torch.nn.functional.embedding(*x),
        (
            torch.full((1, 10), 0, dtype=torch.long),
            torch.rand(10, 10),
        ),
    )
    self.assertEqual(stablehlo.count('stablehlo.composite "odml.embedding_lookup"'), 1)


if __name__ == '__main__':
  unittest.main()
