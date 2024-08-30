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
# Testing model conversion for a few gen-ai models.
import copy

import ai_edge_torch
from ai_edge_torch import config as ai_edge_config
from ai_edge_torch.generative.examples.gemma import gemma, gemma2
from ai_edge_torch.generative.examples.phi2 import phi2
from ai_edge_torch.generative.examples.test_models import toy_model_with_kv_cache  # NOQA
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.testing import model_coverage
import numpy as np
import torch

from absl.testing import absltest as googletest
from tensorflow.lite.python import interpreter


class TestModelConversion(googletest.TestCase):
  """Unit tests that check for model conversion and correctness."""

  def setUp(self):
    super().setUp()
    # Builder function for an Interpreter that supports custom ops.
    self._interpreter_builder = (
        lambda tflite_model: lambda: interpreter.InterpreterWithCustomOps(
            custom_op_registerers=["GenAIOpsRegisterer"],
            model_content=tflite_model,
            experimental_default_delegate_latest_features=True,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_gemma(self):
    config = gemma.get_fake_model_config()
    model = gemma.Gemma(config)

    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10)

    edge_model = ai_edge_torch.convert(model, (tokens, input_pos))
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            model,
            (tokens, input_pos),
            num_valid_inputs=1,
            atol=1e-2,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_gemma2(self):
    config = gemma2.get_fake_model_config()
    model = gemma2.Gemma2(config)
    model.eval()

    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10)

    edge_model = ai_edge_torch.convert(model, (tokens, input_pos))
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    # TODO(b/362840003): debug numerical diff.
    skip_output_check = True
    if not skip_output_check:
      self.assertTrue(
          model_coverage.compare_tflite_torch(
              edge_model,
              model,
              (tokens, input_pos),
              num_valid_inputs=1,
              atol=1e-2,
              rtol=1e-5,
          )
      )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_phi2(self):
    config = phi2.get_fake_model_config()
    pytorch_model = phi2.Phi2(config).eval()

    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10)

    edge_model = ai_edge_torch.convert(pytorch_model, (tokens, input_pos))
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            pytorch_model,
            (tokens, input_pos),
            num_valid_inputs=1,
            atol=1e-3,
            rtol=1e-3,
        )
    )


if __name__ == "__main__":
  googletest.main()
