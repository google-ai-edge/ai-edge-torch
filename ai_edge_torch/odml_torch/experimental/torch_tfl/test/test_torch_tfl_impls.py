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
"""Numerical validation tests for torch ops and Torch-TFL ops."""
from ai_edge_torch import testing
from ai_edge_torch.odml_torch.experimental import torch_tfl
import torch
from torch.utils import _pytree as pytree

from absl.testing import absltest as googletest
from absl.testing import parameterized

export_with_tensor_inputs_only = testing.export_with_tensor_inputs_only


def rnd(dtype, shape, min_v=None, max_v=None):
  """Shortcut for creating a random torch tensor."""
  if dtype in (torch.int32, torch.int64, torch.bool):
    min_v = min_v if min_v else 1
    max_v = max_v if max_v else 10
    return torch.randint(min_v, max_v, shape).to(dtype)
  else:
    min_v = min_v if min_v else 0.0
    max_v = max_v if max_v else 1.0
    return (torch.rand(shape) * (max_v - min_v) + min_v).to(dtype)


class TestTorchTFLImpls(parameterized.TestCase):
  """Numerical validation tests for torch ops and Torch-TFL ops.

  The op test suite is forked from
  ai_edge_torch/odml_torch/test/test_core_aten_ops.py. Eventually, we should
  merge the two test suites.
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _assert_export_and_close(
      self, func, args, kwargs, atol=1e-3, rtol=1e-5, equal_nan=True
  ):
    """Assert func, args, and kwargs can be lowered and pass numerical validation."""
    with self.subTest("torch_eval"):
      expected = func(*args, **kwargs)

      with self.subTest("export_and_decompse"):
        exported_program = export_with_tensor_inputs_only(func, args, kwargs)
        exported_program = exported_program.run_decompositions(
            torch_tfl.decomps
        )

        with self.subTest("decomp_eval"):
          args, kwargs = exported_program.example_inputs
          actual = exported_program.module()(*args, **kwargs)

          with self.subTest("torch_lower_eval_diff:" + str(atol)):
            expected_flat, expected_spec = pytree.tree_flatten(expected)
            actual_flat, actual_spec = pytree.tree_flatten(actual)

            self.assertEqual(expected_spec, actual_spec)
            for v1, v2 in zip(expected_flat, actual_flat):
              torch.testing.assert_close(
                  v1, v2, atol=atol, rtol=rtol, equal_nan=equal_nan
              )

  @parameterized.named_parameters(
      # fmt: off
      # pyformat: disabledef
      ("aten_mm_0", torch.ops.aten.mm.default, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mm_1", torch.ops.aten.mm.default, (rnd(torch.float32, (2, 10)), rnd(torch.float32, (10, 5)),), dict()),
      ("aten_add_Tensor_0", torch.ops.aten.add.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_add_Tensor_1", torch.ops.aten.add.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict(alpha=10)),
      ("aten_sub_Tensor_0", torch.ops.aten.sub.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sub_Tensor_1", torch.ops.aten.sub.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict(alpha=10)),
      ("aten_mul_Tensor_0", torch.ops.aten.mul.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mul_Tensor_1", torch.ops.aten.mul.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_div_Tensor_0", torch.ops.aten.div.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_div_Tensor_1", torch.ops.aten.div.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_gt_Tensor_0", torch.ops.aten.gt.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_gt_Tensor_1", torch.ops.aten.gt.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      # fmt: on
      # pyformat: enable
  )
  def test_op(self, op, args, kwargs):
    self._assert_export_and_close(op, args, kwargs)


if __name__ == "__main__":
  googletest.main()
