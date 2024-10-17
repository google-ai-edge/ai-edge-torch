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

"""A suite of tests to validate the Dynamic Update Slice Custom Op."""

from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import dynamic_update_slice as dus
import torch

from absl.testing import absltest as googletest


def updated_slice_is_the_same(buffer, update, index):
  indexer = [slice(i, i + d) for i, d in zip(index, update.shape)]
  buf = buffer[indexer]
  return torch.allclose(buf, update)


class TestCustomDUS(googletest.TestCase):

  def test_opcheck_dynamic_update_slice(self):
    examples = [
        [
            torch.randn(1, 1280, 4, 64),
            torch.randn([1, 1024, 4, 64]),
            torch.tensor([0, 0, 0, 0], dtype=torch.int),
        ],
        [
            torch.randn(2, 1280, 4, 64),
            torch.randn([2, 1024, 4, 64]),
            torch.tensor([0, 0, 0, 0], dtype=torch.int),
        ],
        [
            torch.randn(2, 256, 4, 64),
            torch.randn([2, 256, 2, 64]),
            torch.tensor([0, 0, 2, 0], dtype=torch.int),
        ],
        [
            torch.randn(2, 256, 4, 64),
            torch.randn([2, 256, 3, 64]),
            torch.tensor([0, 0, 1, 0], dtype=torch.int),
        ],
        [
            torch.randn(6, 8, 32),
            torch.randn([6, 3, 32]),
            torch.tensor([0, 5, 0], dtype=torch.int),
        ],
        [
            torch.randn(8, 32),
            torch.randn([8, 12]),
            torch.tensor([0, 20], dtype=torch.int),
        ],
    ]

    for ex in examples:
      torch.library.opcheck(dus.dynamic_update_slice, ex)
      out = dus.dynamic_update_slice(*ex)
      self.assertTrue(updated_slice_is_the_same(out, ex[1], ex[2]))


if __name__ == "__main__":
  googletest.main()
