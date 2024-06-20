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

import ai_edge_torch


class Identity(torch.nn.Module):

  def forward(self, x):
    return x


class TestToChannelLastIO(unittest.TestCase):
  """Tests to_channel_last_io API and module wrapper."""

  def test_no_transformations(self):
    x = torch.rand(1, 3, 10, 10)
    y = ai_edge_torch.to_channel_last_io(Identity())(x)
    self.assertEqual(y.shape, (1, 3, 10, 10))

  def test_args(self):
    x = torch.rand(1, 10, 10, 3)
    y = ai_edge_torch.to_channel_last_io(Identity(), args=[0])(x)
    self.assertEqual(y.shape, (1, 3, 10, 10))

  def test_outputs(self):
    x = torch.rand(1, 3, 10, 10)
    y = ai_edge_torch.to_channel_last_io(Identity(), outputs=[0])(x)
    self.assertEqual(y.shape, (1, 10, 10, 3))

  def test_args_outputs(self):
    x = torch.rand(1, 10, 10, 3)
    y = ai_edge_torch.to_channel_last_io(Identity(), args=[0], outputs=[0])(x)
    self.assertEqual(y.shape, (1, 10, 10, 3))

  def test_args_outputs_5d(self):
    x = torch.rand(1, 10, 10, 10, 3)
    y = ai_edge_torch.to_channel_last_io(Identity(), args=[0], outputs=[0])(x)
    self.assertEqual(y.shape, (1, 10, 10, 10, 3))

  def test_chained_wrappers(self):
    x = torch.rand(1, 10, 10, 3)

    m = Identity()
    m = ai_edge_torch.to_channel_last_io(m, args=[0])
    m = ai_edge_torch.to_channel_last_io(m, outputs=[0])

    y = m(x)
    self.assertEqual(y.shape, (1, 10, 10, 3))


if __name__ == "__main__":
  unittest.main()
