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

"""Verifies the reauthored PaliGemma 3B model."""

import logging
import pathlib
from absl import app
from absl import flags
from ai_edge_torch.generative.examples.paligemma import paligemma
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import verifier
from PIL import Image
import requests
import torch
import transformers

_IMAGE_URL = flags.DEFINE_string(
    "image_url",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    "The image URI to encode.",
)
_PROMPTS = flags.DEFINE_string(
    "prompts",
    "Caption en",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


class ReauthoredPaliGemmaWrapper(verifier.ReauthoredModelWrapper):
  """Reauthored PaliGemma model wrapper."""

  def _init_kv_cache(self):
    return kv_cache.KVCache.from_model_config(self.model.config.decoder_config)


def main(_):
  checkpoint = "google/paligemma-3b-mix-224"
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = (
      transformers.PaliGemmaForConditionalGeneration.from_pretrained(checkpoint)
  )

  # Locate the cached dir.
  cached_config_file = transformers.utils.cached_file(
      checkpoint, transformers.utils.CONFIG_NAME
  )
  reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = paligemma.build_model(reauthored_checkpoint)

  logging.info("Loading the processor from: %s", checkpoint)
  # It works only when GemmaTokenizerFast is available. In some environments,
  # use_fast=False doeesn't work either if the tokenizer cannot load the
  # sentencepiece model file properly.
  processor = transformers.AutoProcessor.from_pretrained(checkpoint)

  logging.info("Loading the image from: %s", _IMAGE_URL.value)
  image = Image.open(requests.get(_IMAGE_URL.value, stream=True).raw)
  inputs = processor(text=_PROMPTS.value, images=image, return_tensors="pt")

  logging.info("Verifying the reauthored model with model.forward()...")
  logging.info("Forwarding the original model...")
  outputs_original = original_model.forward(
      input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"]
  )
  outputs_original = outputs_original.logits
  logging.info("outputs_original: %s", outputs_original)

  logging.info("Forwarding the reauthored model...")
  wrapped_reauthored_model = ReauthoredPaliGemmaWrapper(reauthored_model)
  outputs_reauthored = wrapped_reauthored_model.forward(
      tokens=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
  )
  logging.info("outputs_reauthored: %s", outputs_reauthored)

  try:
    assert torch.allclose(outputs_original, outputs_reauthored, atol=1e-03)
  except AssertionError as e:
    logging.error("*** FAILED *** verify with forward()")
    raise e
  else:
    logging.info("*** PASSED *** verify with forward()")

  logging.info("Verifying the reauthored model with model.generate()...")
  logging.info("Generating answer with the original model...")
  outputs_original = original_model.generate(
      **inputs, max_new_tokens=_MAX_NEW_TOKENS.value, do_sample=False
  )
  response_original = processor.decode(
      outputs_original[0], skip_special_tokens=True
  )
  logging.info("outputs_from_original_model: [[%s]]", response_original)

  logging.info("Generating answer with the reauthored model...")
  outputs_reauthored = wrapped_reauthored_model.generate(
      prompts=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=_MAX_NEW_TOKENS.value,
  )
  response_reauthored = processor.decode(
      outputs_reauthored[0], skip_special_tokens=True
  )
  logging.info("outputs from reauthored model: [[%s]]", response_reauthored)

  try:
    assert response_original == response_reauthored
  except AssertionError as e:
    logging.error("*** FAILED *** verify with generate()")
    raise e
  else:
    logging.info("*** PASSED *** verify with generate()")


if __name__ == "__main__":
  app.run(main)
