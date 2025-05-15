# Copyright 2025 The AI Edge Torch Authors.
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

"""Verifies the reauthored Qwen 2.5 VL model."""

import logging
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.qwen_vl import qwen_vl
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import transformers_verifier
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
_PROMPTS_WITH_IMAGE = flags.DEFINE_string(
    "prompts_with_image",
    "<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>",
    "The input prompts to generate answers with an image.",
)
_PROMPTS_TEXT_ONLY = flags.DEFINE_multi_string(
    "prompts_text_only",
    "What is the meaning of life?",
    "The input prompts to generate answers only with text.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


class ReauthoredQwenVLWrapper(verifier.ReauthoredModelWrapper):
  """Reauthored Qwen VL model wrapper."""

  def _init_kv_cache(self):
    return kv_cache.KVCache.from_model_config(self.model.config.decoder_config)


def main(_):
  checkpoint = "Qwen/Qwen2.5-VL-3B-Instruct"
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = (
      transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
          checkpoint
      )
  )

  # Locate the cached dir.
  cached_config_file = transformers.utils.cached_file(
      checkpoint, transformers.utils.CONFIG_NAME
  )
  reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = qwen_vl.build_model(str(reauthored_checkpoint))
  wrapped_reauthored_model = ReauthoredQwenVLWrapper(reauthored_model)

  logging.info("Loading the processor from: %s", checkpoint)
  processor = transformers.AutoProcessor.from_pretrained(checkpoint)

  logging.info("Verifying with text-only prompts...")
  verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=wrapped_reauthored_model,
      tokenizer=verifier.TokenizerWrapper(processor.tokenizer),
      generate_prompts=_PROMPTS_TEXT_ONLY.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      atol=1e-04,
  )

  logging.info("Verifying with image input...")
  logging.info("Loading the image from: %s", _IMAGE_URL.value)
  image = Image.open(requests.get(_IMAGE_URL.value, stream=True).raw)
  inputs = processor(
      text=_PROMPTS_WITH_IMAGE.value, images=image, return_tensors="pt"
  )

  logging.info("Verifying the reauthored model with model.forward()...")
  logging.info("Forwarding the original model...")
  outputs_original = original_model.forward(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      image_grid_thw=inputs["image_grid_thw"],
  )
  outputs_original = outputs_original.logits
  logging.info("outputs_original: %s", outputs_original)

  logging.info("Forwarding the reauthored model...")
  grid_thw = inputs["image_grid_thw"].tolist()
  config = reauthored_model.config.image_encoder_config.image_embedding
  reauthored_model.image_encoder.set_image_size(
      (grid_thw[0][1] * config.patch_size, grid_thw[0][2] * config.patch_size)
  )
  outputs_reauthored = wrapped_reauthored_model.forward(
      tokens=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
  )
  logging.info("outputs_reauthored: %s", outputs_reauthored)

  try:
    assert torch.allclose(outputs_original, outputs_reauthored, atol=1e-01)
  except AssertionError as e:
    logging.error("*** FAILED *** verify with forward()")
    raise e
  else:
    logging.info("*** PASSED *** verify with forward()")

  logging.info("Verifying the reauthored model with model.generate()...")
  logging.info("Generating answer with the original model...")
  outputs_original = original_model.generate(
      **inputs, max_new_tokens=_MAX_NEW_TOKENS.value
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
