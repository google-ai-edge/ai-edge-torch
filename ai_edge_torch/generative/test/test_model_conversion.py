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
  def test_toy_model_with_kv_cache(self):
    config = toy_model_with_kv_cache.get_model_config()
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKV(config).eval()
    idx, input_pos = torch.tensor([[1]], dtype=torch.long), torch.tensor(
        [10], dtype=torch.int64
    )

    edge_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            pytorch_model,
            (idx, input_pos),
            num_valid_inputs=1,
            atol=1e-5,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_toy_model_with_kv_cache_with_hlfb(self):
    config = toy_model_with_kv_cache.get_model_config()
    config.enable_hlfb = True
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKV(config).eval()
    idx, input_pos = torch.tensor([[1]], dtype=torch.long), torch.tensor(
        [10], dtype=torch.int64
    )

    edge_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            pytorch_model,
            (idx, input_pos),
            num_valid_inputs=1,
            atol=1e-5,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_tiny_llama_multisig(self):
    config = tiny_llama.get_fake_model_config()
    pytorch_model = tiny_llama.TinyLLamma(config).eval()

    # prefill
    seq_len = 10
    prefill_tokens = torch.full((1, seq_len), 0, dtype=torch.long, device="cpu")
    prompt_token = torch.from_numpy(np.array([1, 2, 3, 4]))
    prefill_tokens[0, : len(prompt_token)] = prompt_token
    prefill_input_pos = torch.arange(0, seq_len)

    # decode
    decode_token = torch.tensor([[1]], dtype=torch.long)
    decode_input_pos = torch.tensor([5], dtype=torch.int64)

    edge_model = (
        ai_edge_torch.signature(
            "prefill", pytorch_model, (prefill_tokens, prefill_input_pos)
        )
        .signature("decode", pytorch_model, (decode_token, decode_input_pos))
        .convert()
    )
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    copied_model = copy.deepcopy(pytorch_model)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model,
            pytorch_model,
            (prefill_tokens, prefill_input_pos),
            signature_name="prefill",
            num_valid_inputs=1,
        )
    )

    # TODO(b/362840003): figure why this decode output has big numerical diff.
    skip_output_check = True
    if not skip_output_check:
      self.assertTrue(
          model_coverage.compare_tflite_torch(
              edge_model,
              copied_model,
              (decode_token, decode_input_pos),
              signature_name="decode",
              num_valid_inputs=1,
          )
      )


if __name__ == "__main__":
  googletest.main()
