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

from ai_edge_torch.generative.layers import attention_utils
import torch

from absl.testing import absltest as googletest


class AttentionUtilsTest(googletest.TestCase):

  def test_build_causal_mask_cache(self):
    mask = attention_utils.build_causal_mask_cache(3)
    self.assertEqual(mask.shape, (1, 1, 3, 3))
    self.assertEqual(mask[0, 0, 0, 0], 0)
    self.assertEqual(mask[0, 0, 0, 1], float("-inf"))
    self.assertEqual(mask[0, 0, 0, 2], float("-inf"))
    self.assertEqual(mask[0, 0, 1, 0], 0)
    self.assertEqual(mask[0, 0, 1, 1], 0)
    self.assertEqual(mask[0, 0, 1, 2], float("-inf"))
    self.assertEqual(mask[0, 0, 2, 0], 0)
    self.assertEqual(mask[0, 0, 2, 1], 0)
    self.assertEqual(mask[0, 0, 2, 2], 0)

  def test_build_sliding_window_mask_cache(self):
    mask = attention_utils.build_sliding_window_mask_cache(3, 2)
    self.assertEqual(mask.shape, (1, 1, 3, 3))
    self.assertEqual(mask[0, 0, 0, 0], 0)
    self.assertEqual(mask[0, 0, 0, 1], float("-inf"))
    self.assertEqual(mask[0, 0, 0, 2], float("-inf"))
    self.assertEqual(mask[0, 0, 1, 0], 0)
    self.assertEqual(mask[0, 0, 1, 1], 0)
    self.assertEqual(mask[0, 0, 1, 2], float("-inf"))
    self.assertEqual(mask[0, 0, 2, 0], float("-inf"))
    self.assertEqual(mask[0, 0, 2, 1], 0)
    self.assertEqual(mask[0, 0, 2, 2], 0)

  def test_build_relative_position_buckets(self):
    buckets = attention_utils.build_relative_position_buckets(
        query_length=3, key_length=3, bidirectional=True, num_buckets=4
    )
    print(buckets)
    self.assertEqual(buckets.shape, (1, 1, 3, 3))
    self.assertTrue(
        torch.equal(
            buckets, torch.tensor([[[[0, 3, 3], [1, 0, 3], [1, 1, 0]]]])
        )
    )


if __name__ == "__main__":
  googletest.main()
