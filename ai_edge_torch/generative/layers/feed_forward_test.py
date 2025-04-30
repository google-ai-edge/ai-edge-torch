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

from ai_edge_torch.generative.layers import feed_forward
import torch
import torch.nn.functional as F
from absl.testing import absltest as googletest


class FeedForwardTest(googletest.TestCase):

  def test_sequential_feed_forward(self):
    ff = feed_forward.SequentialFeedForward(
        dim=10,
        hidden_dim=10,
        activation=F.silu,
        use_bias=True,
        use_glu=False,
        pre_ff_norm=torch.nn.Identity(),
        post_ff_norm=torch.nn.Identity(),
    )
    x = torch.ones((1, 10))
    out = ff(x)
    self.assertEqual(out.shape, (1, 10))

  def test_gated_feed_forward(self):
    ff = feed_forward.GatedFeedForward(
        dim=10,
        hidden_dim=10,
        activation=F.silu,
        use_bias=True,
        use_glu=False,
        pre_ff_norm=torch.nn.Identity(),
        post_ff_norm=torch.nn.Identity(),
    )
    x = torch.ones((1, 10))
    out = ff(x)
    self.assertEqual(out.shape, (1, 10))


if __name__ == "__main__":
  googletest.main()
