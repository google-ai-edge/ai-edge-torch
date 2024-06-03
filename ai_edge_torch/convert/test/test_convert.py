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


import os
import tempfile
import unittest

import torch
import torchvision

import ai_edge_torch
from ai_edge_torch.convert import conversion_utils as cutils
from ai_edge_torch.testing import model_coverage


class TestConvert(unittest.TestCase):
  """Tests conversion of various modules."""

  def setUp(self):
    torch.manual_seed(0)

  def test_convert_add(self):
    """Tests conversion of a simple Add module."""

    class Add(torch.nn.Module):

      def forward(self, a, b):
        return a + b

    args = (
        torch.randn((5, 10)),
        torch.randn((5, 10)),
    )
    torch_module = Add().eval()
    edge_model = ai_edge_torch.convert(torch_module, args)

    self.assertTrue(model_coverage.compare_tflite_torch(edge_model, torch_module, args))

  def test_convert_dot_add(self):
    class DotAdd(torch.nn.Module):
      """Tests conversion of a matrix multiplication followed by an add."""

      def forward(self, a, b, c):
        return a @ b + c

    args = (
        torch.randn((5, 10)),
        torch.randn((10, 5)),
        torch.randn((5, 5)),
    )
    torch_module = DotAdd().eval()
    edge_model = ai_edge_torch.convert(torch_module, args)

    self.assertTrue(model_coverage.compare_tflite_torch(edge_model, torch_module, args))

  def test_convert_resnet18(self):
    args = (torch.randn(4, 3, 224, 224),)
    torch_module = torchvision.models.resnet18().eval()
    edge_model = ai_edge_torch.convert(torch_module, args)

    self.assertTrue(model_coverage.compare_tflite_torch(edge_model, torch_module, args))

  def test_signature_args_ordering(self):
    """Tests conversion of a model with more than 10 arguments."""

    class AddChainWith11Args(torch.nn.Module):

      def forward(
          self,
          arg0: "f32[64]",
          arg1: "f32[64]",
          arg2: "f32[64]",
          arg3: "f32[64]",
          arg4: "f32[64]",
          arg5: "f32[64]",
          arg6: "f32[64]",
          arg7: "f32[64]",
          arg8: "f32[64]",
          arg9: "f32[64]",
          arg10: "f32[64]",
      ):
        add0 = torch.add(arg0, arg1)
        add1 = torch.add(add0, arg2)
        add2 = torch.add(add1, arg3)
        add3 = torch.add(add2, arg4)
        add4 = torch.add(add3, arg5)
        add5 = torch.add(add4, arg6)
        add6 = torch.add(add5, arg7)
        add7 = torch.add(add6, arg8)
        add8 = torch.add(add7, arg9)
        add9 = torch.add(add8, arg10)
        return add9

    sample_input = lambda: (
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
    )
    torch_model = AddChainWith11Args().eval()
    edge_model = ai_edge_torch.convert(torch_model, sample_input())

    result = model_coverage.compare_tflite_torch(
        edge_model, torch_model, sample_input, num_valid_inputs=10
    )
    self.assertTrue(result)

  def test_multi_output_model(self):
    """Tests conversion of a model that returns multiple outputs."""

    class BasicAddModelWithMultipleOutputs(torch.nn.Module):

      def forward(self, arg0, arg1):
        add0 = arg0 + arg1
        mul0 = arg0 * arg1
        return add0, mul0

    sample_input = (
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
    )

    torch_model = BasicAddModelWithMultipleOutputs().eval()
    edge_model = ai_edge_torch.convert(torch_model, sample_input)

    result = model_coverage.compare_tflite_torch(edge_model, torch_model, sample_input)
    self.assertTrue(result)

  def test_12_outputs_model(self):
    """Tests conversion of a model that returns multiple outputs."""

    class BasicAddModelWithMultipleOutputs(torch.nn.Module):

      def forward(self, arg0, arg1):
        add0 = arg0 + arg1
        mul0 = arg0 * arg1
        add1 = add0 + mul0
        mul1 = add0 * mul0
        add2 = add1 + mul1
        mul2 = add1 * mul1
        add3 = add2 + mul2
        mul3 = add2 * mul2
        add4 = add3 + mul3
        mul4 = add3 * mul3
        add5 = add4 + mul4
        mul5 = add4 * mul4

        return (
            add0,
            mul0,
            add1,
            mul1,
            add2,
            mul2,
            add3,
            mul3,
            add4,
            mul4,
            add5,
            mul5,
        )

    sample_input = (
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
    )

    torch_model = BasicAddModelWithMultipleOutputs().eval()
    edge_model = ai_edge_torch.convert(torch_model, sample_input)

    result = model_coverage.compare_tflite_torch(edge_model, torch_model, sample_input)
    self.assertTrue(result)

  def test_apply_tfl_backdoor_flags(self):
    """Tests if _apply_tfl_backdoor_flags correctly sets the values in a Converter object."""

    class MockConverterInternalObject:

      def __init__(self):
        self.subkey2 = "original_subvalue2"

    class MockConverter:

      def __init__(self):
        self.key1 = "original_value1"
        self.key2 = MockConverterInternalObject()

    mock_converter = MockConverter()
    flags = {"key1": "new_value1", "key2": {"subkey2": "new_subvalue2"}}
    cutils._apply_tfl_backdoor_flags(mock_converter, flags)

    self.assertTrue(flags["key1"], "new_value1")
    self.assertTrue(flags["key2"]["subkey2"], "new_subvalue2")

  def test_convert_add_backdoor_flags(self):
    """Tests conversion of an add module setting a tflite converter flag."""

    class Add(torch.nn.Module):

      def forward(self, a, b):
        return a + b

    args = (
        torch.randn((5, 10)),
        torch.randn((5, 10)),
    )
    torch_module = Add().eval()

    with tempfile.TemporaryDirectory() as tmp_dir_path:
      ir_dump_path = os.path.join(
          tmp_dir_path, "test_convert_add_backdoor_flags_mlir_dump"
      )
      ai_edge_torch.convert(
          torch_module, args, _ai_edge_converter_flags={"ir_dump_dir": ir_dump_path}
      )
      self.assertTrue(os.path.isdir(ir_dump_path))

  def test_convert_model_with_dynamic_batch(self):
    """
    Test converting a simple model with dynamic batch size.
    """

    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.ones((10, 10)) * 2.7

      def forward(self, x, y):
        return x + y + self.w

    sample_input = (torch.randn(4, 3, 10, 10), torch.randn(4, 3, 10, 10))
    batch = torch.export.Dim("batch")
    dynamic_shapes = ({0: batch}, {0: batch})

    model = SampleModel().eval()
    edge_model = ai_edge_torch.convert(
        model, sample_input, dynamic_shapes=dynamic_shapes
    )

    for batch_size in [2, 4, 10]:
      validate_input = (
          torch.randn(batch_size, 3, 10, 10),
          torch.randn(batch_size, 3, 10, 10),
      )
      self.assertTrue(
          model_coverage.compare_tflite_torch(edge_model, model, validate_input)
      )


if __name__ == "__main__":
  unittest.main()
