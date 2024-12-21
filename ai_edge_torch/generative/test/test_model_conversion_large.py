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
from ai_edge_torch.generative.examples.amd_llama_135m import amd_llama_135m
from ai_edge_torch.generative.examples.gemma import gemma1
from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.examples.llama import llama
from ai_edge_torch.generative.examples.openelm import openelm
from ai_edge_torch.generative.examples.paligemma import decoder
from ai_edge_torch.generative.examples.paligemma import decoder2
from ai_edge_torch.generative.examples.paligemma import paligemma
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.examples.phi import phi3
from ai_edge_torch.generative.examples.qwen import qwen
from ai_edge_torch.generative.examples.smollm import smollm
from ai_edge_torch.generative.examples.stable_diffusion import clip as sd_clip
from ai_edge_torch.generative.examples.stable_diffusion import decoder as sd_decoder
from ai_edge_torch.generative.examples.stable_diffusion import diffusion as sd_diffusion
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.test import utils as test_utils
import numpy as np
import torch

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter


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
    # Default cache_size_limit, 8 is hit and aborts often when the tests are
    # running all together. Doubles it to avoid abortion.
    torch._dynamo.config.cache_size_limit = 16
    np.random.seed(1234)  # Make np.random deterministic.

  def _test_model(self, config, model, signature_name, atol, rtol):
    seq_len = 10
    tokens = torch.zeros((1, seq_len), dtype=torch.int, device="cpu")
    input_pos = torch.arange(0, seq_len, dtype=torch.int)
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

    tokens = torch.arange(1, seq_len + 1, dtype=torch.int).unsqueeze(0)
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
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_gemma1(self):
    config = gemma1.get_fake_model_config()
    pytorch_model = gemma1.Gemma1(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_gemma2(self):
    config = gemma2.get_fake_model_config()
    pytorch_model = gemma2.Gemma2(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_llama(self):
    config = llama.get_fake_model_config()
    pytorch_model = llama.Llama(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_phi2(self):
    config = phi2.get_fake_model_config()
    pytorch_model = phi2.Phi2(config).eval()
    # Phi-2 logits are very big, so we need a larger absolute tolerance.
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_phi3(self):
    config = phi3.get_fake_model_config()
    pytorch_model = phi3.Phi3_5Mini(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-5, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_smollm(self):
    config = smollm.get_fake_model_config()
    pytorch_model = smollm.SmolLM(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_openelm(self):
    config = openelm.get_fake_model_config()
    pytorch_model = openelm.OpenELM(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_qwen(self):
    config = qwen.get_fake_model_config()
    pytorch_model = qwen.Qwen(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_amd_llama_135m(self):
    config = amd_llama_135m.get_fake_model_config()
    pytorch_model = amd_llama_135m.AmdLlama(config).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-5, rtol=1e-5)

  def _test_paligemma_model(self, decoder_class, decoder_config, atol, rtol):
    config = paligemma.get_fake_model_config(decoder_config)
    pytorch_model = paligemma.PaliGemma(config, decoder_class).eval()

    image_embedding_config = config.image_encoder_config.image_embedding
    num_patches = (
        image_embedding_config.image_size // image_embedding_config.patch_size
    ) ** 2

    # Make sure the token size is longer than the number of image patches.
    seq_len = num_patches + 10
    tokens = torch.zeros((1, seq_len), dtype=torch.int, device="cpu")
    input_pos = torch.arange(0, seq_len, dtype=torch.int)
    kv = kv_cache.KVCache.from_model_config(config.decoder_config)
    pixel_values = torch.zeros((1, 3, 8, 8), dtype=torch.float32, device="cpu")

    edge_model = ai_edge_torch.signature(
        "prefill_pixel",
        pytorch_model,
        sample_kwargs={
            "tokens": tokens,
            "input_pos": input_pos,
            "kv_cache": kv,
            "pixel_values": pixel_values,
        },
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )

    tokens = torch.arange(1, seq_len + 1, dtype=torch.int).unsqueeze(0)
    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            pytorch_model,
            tokens,
            input_pos,
            kv,
            pixel_values=pixel_values,
            signature_name="prefill_pixel",
            atol=atol,
            rtol=rtol,
        )
    )

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def disabled_test_paligemma1(self):
    self._test_paligemma_model(
        decoder.Decoder, decoder.get_fake_decoder_config, atol=1e-3, rtol=1e-5
    )

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def disabled_test_paligemma2(self):
    self._test_paligemma_model(
        decoder2.Decoder2,
        decoder2.get_fake_decoder2_config,
        atol=1e-3,
        rtol=1e-5,
    )

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_stable_diffusion_clip(self):
    config = sd_clip.get_fake_model_config()
    prompt_tokens = torch.from_numpy(
        np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    )

    pytorch_model = sd_clip.CLIP(config).eval()
    torch_output = pytorch_model(prompt_tokens)

    edge_model = ai_edge_torch.signature(
        "encode", pytorch_model, (prompt_tokens,)
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )
    edge_output = edge_model(
        prompt_tokens.numpy(),
        signature_name="encode",
    )
    self.assertTrue(
        test_utils.compare_logits(
            edge_output,
            torch_output.detach().numpy(),
            atol=1e-4,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_stable_diffusion_diffusion(self):
    config = sd_diffusion.get_fake_model_config(2)
    # Reduce stddev(scale) of input values to avoid too big output logits which
    # fails comparisons with reasonable tolerances.
    latents = torch.from_numpy(
        np.random.normal(size=(2, 4, 8, 8), scale=0.1).astype(np.float32)
    )
    context = torch.from_numpy(
        np.random.normal(size=(2, 4, 4), scale=0.1).astype(np.float32)
    )
    time_embedding = torch.from_numpy(
        np.random.normal(size=(2, 2), scale=0.1).astype(np.float32)
    )

    pytorch_model = sd_diffusion.Diffusion(config).eval()
    torch_output = pytorch_model(latents, context, time_embedding)

    edge_model = ai_edge_torch.signature(
        "diffusion", pytorch_model, (latents, context, time_embedding)
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )
    edge_output = edge_model(
        latents.numpy(),
        context.numpy(),
        time_embedding.numpy(),
        signature_name="diffusion",
    )
    self.assertTrue(
        test_utils.compare_logits(
            edge_output,
            torch_output.detach().numpy(),
            atol=1e-4,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(
      ai_edge_torch.config.in_oss,
      reason="tests with custom ops are not supported in oss",
  )
  def test_stable_diffusion_decoder(self):
    config = sd_decoder.get_fake_model_config()
    # Reduce stddev(scale) of input values to avoid too big output logits which
    # fails comparisons with reasonable tolerances.
    latents = torch.from_numpy(
        np.random.normal(size=(1, 4, 64, 64), scale=0.1).astype(np.float32)
    )

    pytorch_model = sd_decoder.Decoder(config).eval()
    torch_output = pytorch_model(latents)

    edge_model = ai_edge_torch.signature(
        "decode", pytorch_model, (latents,)
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.tflite_model())
    )
    edge_output = edge_model(
        latents.numpy(),
        signature_name="decode",
    )
    self.assertTrue(
        test_utils.compare_logits(
            edge_output,
            torch_output.detach().numpy(),
            atol=1e-3,
            rtol=1e-5,
        )
    )


if __name__ == "__main__":
  googletest.main()
