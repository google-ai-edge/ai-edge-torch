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
from ai_edge_torch.generative.examples.test_models import toy_model_with_kv_cache
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
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

  def _test_model_with_kv_cache(self, config, pytorch_model):
    tokens, input_pos = torch.tensor([[1]], dtype=torch.int), torch.tensor(
        [10], dtype=torch.int
    )
    kv = kv_cache.KVCache.from_model_config(config)

    edge_model = ai_edge_torch.convert(
        pytorch_model,
        sample_kwargs={
            "tokens": tokens,
            "input_pos": input_pos,
            "kv_cache": kv,
        },
    )
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            pytorch_model,
            tokens,
            input_pos,
            kv,
            signature_name="serving_default",
            atol=1e-5,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_toy_model_with_kv_cache(self):
    config = toy_model_with_kv_cache.get_model_config()
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKVCache(config).eval()
    self._test_model_with_kv_cache(config, pytorch_model)

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_toy_model_with_kv_cache_with_hlfb(self):
    config = toy_model_with_kv_cache.get_model_config()
    config.enable_hlfb = True
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKVCache(config).eval()
    self._test_model_with_kv_cache(config, pytorch_model)

  def _test_multisig_model(self, config, pytorch_model, atol, rtol):
    # prefill
    seq_len = 10
    prefill_tokens = torch.full((1, seq_len), 0, dtype=torch.int, device="cpu")
    prompt_token = torch.from_numpy(np.array([1, 2, 3, 4]))
    prefill_tokens[0, : len(prompt_token)] = prompt_token
    prefill_input_pos = torch.arange(0, seq_len, dtype=torch.int)

    # decode
    decode_token = torch.tensor([[1]], dtype=torch.int)
    decode_input_pos = torch.tensor([5], dtype=torch.int)

    kv = kv_cache.KVCache.from_model_config(config)

    edge_model = (
        ai_edge_torch.signature(
            "prefill",
            pytorch_model,
            sample_kwargs={
                "tokens": prefill_tokens,
                "input_pos": prefill_input_pos,
                "kv_cache": kv,
            },
        )
        .signature(
            "decode",
            pytorch_model,
            sample_kwargs={
                "tokens": decode_token,
                "input_pos": decode_input_pos,
                "kv_cache": kv,
            },
        )
        .convert()
    )
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            pytorch_model,
            prefill_tokens,
            prefill_input_pos,
            kv,
            signature_name="prefill",
            atol=atol,
            rtol=atol,
        )
    )

    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            pytorch_model,
            decode_token,
            decode_input_pos,
            kv,
            signature_name="decode",
            atol=atol,
            rtol=atol,
        )
    )

  @googletest.skipIf(
      ai_edge_config.Config.use_torch_xla,
      reason="tests with custom ops are not supported on oss",
  )
  def test_tiny_llama_multisig(self):
    config = tiny_llama.get_fake_model_config()
    pytorch_model = tiny_llama.TinyLlama(config).eval()
    self._test_multisig_model(config, pytorch_model, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  googletest.main()
