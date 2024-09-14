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

"""Testing model conversion for a few gen-ai models."""

import ai_edge_torch
from ai_edge_torch import config as ai_edge_config
from ai_edge_torch.generative.examples.gemma import gemma
from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.examples.openelm import openelm
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.examples.smollm import smollm
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.test import utils as test_utils
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

  def _test_model(self, config, model, signature_name, atol, rtol):
    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.int, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10, dtype=torch.int)
    kv = kv_cache.KVCache.from_model_config(config)

    edge_model = ai_edge_torch.signature(
        signature_name,
        model,
        sample_kwargs={
            "tokens": tokens,
            "input_pos": input_pos,
            "kv_cache": kv,
        },
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            model,
            tokens,
            input_pos,
            kv,
            signature_name=signature_name,
            atol=atol,
            rtol=rtol,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_gemma(self):
    config = gemma.get_fake_model_config()
    pytorch_model = gemma.Gemma(config).eval()
    self._test_model(
        config, pytorch_model, "serving_default", atol=1e-2, rtol=1e-5
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_gemma2(self):
    config = gemma2.get_fake_model_config()
    pytorch_model = gemma2.Gemma2(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-1, rtol=1e-3)

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_phi2(self):
    config = phi2.get_fake_model_config()
    pytorch_model = phi2.Phi2(config).eval()
    self._test_model(
        config, pytorch_model, "serving_default", atol=1e-3, rtol=1e-3
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_smollm(self):
    config = smollm.get_fake_model_config()
    pytorch_model = smollm.SmolLM(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_openelm(self):
    config = openelm.get_fake_model_config()
    pytorch_model = openelm.OpenELM(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
  googletest.main()
