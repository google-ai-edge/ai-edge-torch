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
import os
import tempfile
import unittest

import numpy as np
import torch

import ai_edge_torch
from ai_edge_torch.generative.examples.gemma import gemma
from ai_edge_torch.generative.examples.phi2 import phi2
from ai_edge_torch.generative.examples.test_models import toy_model_with_kv_cache  # NOQA
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.testing import model_coverage


class TestModelConversion(unittest.TestCase):
  """Unit tests that check for model conversion and correctness."""

  def test_toy_model_with_kv_cache(self):
    config = toy_model_with_kv_cache.get_model_config()
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKV(config)
    idx, input_pos = torch.tensor([[1]], dtype=torch.long), torch.tensor(
        [10], dtype=torch.int64
    )

    edge_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
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

  def test_toy_model_with_multi_batches(self):
    config = toy_model_with_kv_cache.get_model_config()
    config.batch_size = 2
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKV(config)
    idx, input_pos = torch.tensor([[1], [2]], dtype=torch.long), torch.tensor(
        [10], dtype=torch.int64
    )

    edge_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
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

  def test_toy_model_with_kv_cache_with_hlfb(self):
    config = toy_model_with_kv_cache.get_model_config()
    config.enable_hlfb = True
    pytorch_model = toy_model_with_kv_cache.ToyModelWithKV(config)
    idx, input_pos = torch.tensor([[1]], dtype=torch.long), torch.tensor(
        [10], dtype=torch.int64
    )

    edge_model = ai_edge_torch.convert(pytorch_model, (idx, input_pos))

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
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

  def test_tiny_llama(self):
    self.skipTest("b/338288901")
    config = tiny_llama.get_fake_model_config_for_test()
    pytorch_model = tiny_llama.TinyLLamma(config)

    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10)

    edge_model = ai_edge_torch.convert(pytorch_model, (tokens, input_pos))

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
      self.assertTrue(
          model_coverage.compare_tflite_torch(
              edge_model,
              pytorch_model,
              (tokens, input_pos),
              num_valid_inputs=1,
              atol=1e-5,
              rtol=1e-5,
          )
      )

  def test_tiny_llama_multisig(self):
    config = tiny_llama.get_fake_model_config_for_test()
    pytorch_model = tiny_llama.TinyLLamma(config)

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

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
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

      self.assertTrue(
          model_coverage.compare_tflite_torch(
              edge_model,
              copied_model,
              (decode_token, decode_input_pos),
              signature_name="decode",
              num_valid_inputs=1,
          )
      )

  def test_gemma(self):
    self.skipTest("b/338288901")
    config = gemma.get_fake_model_config_2b_for_test()
    model = gemma.Gemma(config)

    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10)

    edge_model = ai_edge_torch.convert(model, (tokens, input_pos))

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
      # TODO(talumbau, haoliang): debug numerical diff.
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

  def test_phi2(self):
    self.skipTest("b/338288901")
    config = phi2.get_fake_model_config_for_test()
    pytorch_model = phi2.Phi2(config)

    idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
    tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
    tokens[0, :4] = idx
    input_pos = torch.arange(0, 10)

    edge_model = ai_edge_torch.convert(pytorch_model, (tokens, input_pos))

    # TODO(b/338288901): re-enable test to check output tensors.
    skip_output_check = True
    if skip_output_check is False:
      self.assertTrue(
          model_coverage.compare_tflite_torch(
              edge_model,
              pytorch_model,
              (tokens, input_pos),
              num_valid_inputs=1,
              atol=1e-5,
              rtol=1e-5,
          )
      )


if __name__ == "__main__":
  unittest.main()
