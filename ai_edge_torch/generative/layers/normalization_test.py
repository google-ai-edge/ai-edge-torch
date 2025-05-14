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
"""Tests for normalization layers."""

from ai_edge_torch.generative.layers import normalization
import torch
from absl.testing import absltest as googletest
from absl.testing import parameterized


class NormalizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="rms_norm_test_1",
          model_dim=10,
          with_scale=False,
          scale_shift=1.0,
          enable_hlfb=False,
          expected_values=torch.ones((10,), dtype=torch.float32),
      ),
      dict(
          testcase_name="rms_norm_test_2",
          model_dim=10,
          with_scale=True,
          scale_shift=2.0,
          enable_hlfb=False,
          expected_values=torch.ones((10,), dtype=torch.float32) * 2.0,
      ),
      dict(
          testcase_name="rms_norm_test_3",
          model_dim=10,
          with_scale=True,
          scale_shift=2.0,
          enable_hlfb=True,
          expected_values=torch.ones((10,), dtype=torch.float32) * 2.0,
      ),
  )
  def test_rms_norm(
      self,
      model_dim: int,
      with_scale: bool,
      scale_shift: float,
      enable_hlfb: bool,
      expected_values: torch.Tensor,
  ):
    rms_norm = normalization.RMSNorm(
        dim=model_dim,
        with_scale=with_scale,
        scale_shift=scale_shift,
        enable_hlfb=enable_hlfb,
    )

    x = torch.ones((model_dim,), dtype=torch.float32)
    out = rms_norm(x)
    self.assertEqual(out.shape, (model_dim,))
    self.assertTrue(torch.allclose(out, expected_values))


if __name__ == "__main__":
  googletest.main()
