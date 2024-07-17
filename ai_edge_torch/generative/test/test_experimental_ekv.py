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
# A suite of tests to validate experimental external KV Cache models.

import unittest

from ai_edge_torch.generative.examples.experimental.gemma import gemma
from ai_edge_torch.generative.examples.experimental.phi import phi2
from ai_edge_torch.generative.examples.experimental.tiny_llama import tiny_llama  # NOQA


class TestEKVModels(unittest.TestCase):

  def test_can_build_gemma(self):
    gemma.define_and_run_2b(checkpoint_path=None)

  def test_can_build_phi2(self):
    phi2.define_and_run(checkpoint_path=None)

  def test_can_build_tinyllama(self):
    tiny_llama.define_and_run(checkpoint_path=None)


if __name__ == "__main__":
  unittest.main()
