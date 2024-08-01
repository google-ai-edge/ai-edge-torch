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

import ai_edge_torch
from ai_edge_torch.testing import model_coverage
import numpy.testing
import torch
import torchvision


class TestSerialization(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)

  def test_read_write(self):
    """
    (1) Creates an ai_edge_torch model from a torch model
    (2) Saves and then loads the model
    (3) Checks to make sure the model is still runnable and produces the right results.
    """
    resnet18 = torchvision.models.resnet18().eval()
    sample_input = (torch.randn(4, 3, 224, 224),)

    edge_model = ai_edge_torch.convert(resnet18, sample_input)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
      edge_model.export(os.path.join(tmp_dir_name, "test.model"))
      loaded_model = ai_edge_torch.load(
          os.path.join(tmp_dir_name, "test.model")
      )

    result = model_coverage.compare_tflite_torch(
        loaded_model, resnet18, sample_input
    )
    self.assertTrue(result)

  def test_wrong_model_raises(self):
    """Checks if the right exception is raised if the model is not deserializable."""
    with tempfile.NamedTemporaryFile() as fp:
      fp.write(b"dummy data")

      with self.assertRaises(ValueError):
        ai_edge_torch.load(fp.name)


if __name__ == "__main__":
  unittest.main()
