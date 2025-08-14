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
"""Verifies the reauthored SmolVLM2 Image Encoder model."""

import logging

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.smolvlm2 import smolvlm2
from ai_edge_torch.generative.examples.smolvlm2 import vision_encoder
from PIL import Image
import requests
import torch
import transformers

_IMAGE_URL = flags.DEFINE_string(
    "image_url",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
    "The image URI to encode.",
)

_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "The checkpoint to verify.",
)

_REAUTHORTHED_CHECKPOINT = flags.DEFINE_string(
    "pretrained_weights",
    None,
    "The path to the model's pretrained weights.",
)


def main(_):
  checkpoint = _CHECKPOINT.value
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = transformers.AutoModelForImageTextToText.from_pretrained(
      checkpoint
  )
  original_model = original_model.eval().model

  logging.info("Building the reauthored checkpoint from: %s", checkpoint)
  reauthored_checkpoint = _REAUTHORTHED_CHECKPOINT.value
  if reauthored_checkpoint is None:
    raise ValueError("reauthored_checkpoint is required.")

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = vision_encoder.build_image_encoder(reauthored_checkpoint)

  logging.info("Loading the tokenizer from: %s", checkpoint)
  processor = transformers.AutoProcessor.from_pretrained(checkpoint)

  logging.info("Loading the image from: %s", _IMAGE_URL.value)
  image = Image.open(requests.get(_IMAGE_URL.value, stream=True).raw)
  pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

  logging.info("Forwarding the original model...")
  outputs_original = original_model.get_image_features(pixel_values)
  logging.info("outputs_original's shape: %s", outputs_original.shape)

  pixel_values = pixel_values.reshape(
      pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
  )
  logging.info("Forwarding the reauthored model...")
  outputs_reauthored = reauthored_model.forward(
      pixel_values=pixel_values
  )
  logging.info("outputs_reauthored's shape: %s", outputs_reauthored.shape)

  try:
    assert torch.allclose(
        outputs_original, outputs_reauthored, atol=1e-03, rtol=1e-04
    )
  except AssertionError as e:
    logging.error("*** FAILED *** verify with an image")
    raise e
  else:
    logging.info("*** PASSED *** verify with an image")


if __name__ == "__main__":
  app.run(main)
