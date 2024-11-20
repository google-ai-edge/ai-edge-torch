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

"""Verifies the reauthored image encoder of PaliGemma 3B model."""

import logging
import pathlib
from absl import app
from absl import flags
from ai_edge_torch.generative.examples.paligemma import image_encoder
from PIL import Image
import requests
import torch
import transformers

_IMAGE_URL = flags.DEFINE_string(
    "image_url",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    "The image URI to encode.",
)


def main(_):
  checkpoint = "google/paligemma-3b-mix-224"
  logging.info("Loading the original model from: %s", checkpoint)
  original_full_model = (
      transformers.PaliGemmaForConditionalGeneration.from_pretrained(checkpoint)
  )
  original_vision_model = original_full_model.eval().vision_tower

  # Locate the cached dir.
  cached_config_file = transformers.utils.cached_file(
      checkpoint, transformers.utils.CONFIG_NAME
  )
  reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = image_encoder.build_image_encoder(reauthored_checkpoint)

  logging.info("Loading the processor from: %s", checkpoint)
  # It works only when GemmaTokenizerFast is available. In some environments,
  # use_fast=False doeesn't work either if the tokenizer cannot load the
  # sentencepiece model file properly.
  processor = transformers.AutoProcessor.from_pretrained(checkpoint)

  logging.info("Loading the image from: %s", _IMAGE_URL.value)
  image = Image.open(requests.get(_IMAGE_URL.value, stream=True).raw)
  pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

  logging.info("Forwarding the original model...")
  outputs_original = original_vision_model.forward(pixel_values=pixel_values)
  outputs_original = outputs_original.last_hidden_state
  logging.info("outputs_original: %s", outputs_original)

  logging.info("Forwarding the reauthored model...")
  outputs_reauthored = reauthored_model.forward(pixel_values=pixel_values)
  logging.info("outputs_reauthored: %s", outputs_reauthored)

  try:
    assert torch.allclose(
        outputs_original, outputs_reauthored, atol=1e-04, rtol=1e-04
    )
  except AssertionError as e:
    logging.error("*** FAILED *** verify with an image")
    raise e
  else:
    logging.info("*** PASSED *** verify with an image")


if __name__ == "__main__":
  app.run(main)
