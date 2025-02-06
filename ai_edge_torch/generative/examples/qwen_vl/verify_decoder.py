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

"""Verifies the reauthored decoder of Qwen 2.5 VL 3B models."""

import logging
import pathlib

from absl import app
from ai_edge_torch.generative.examples.qwen_vl import decoder
from ai_edge_torch.generative.utilities import verifier
import torch
import transformers


class DecoderWrapper(verifier.ModelWrapper):
  """Wraps the decoder of Qwen 2.5 VL models for verification."""

  def __init__(self, model: torch.nn.Module, lm_head: torch.nn.Module):
    super().__init__(model)
    self.lm_head = lm_head

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    output = self.model.forward(tokens)
    return self.lm_head(output["last_hidden_state"])


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
  reauthored_model = decoder.build_decoder(reauthored_checkpoint)

  # Verify the reauthored model only with input IDs because the original decoder
  # does not support generate() with prompts.
  input_ids = [1, 2, 3, 4]
  try:
    verifier.verify_with_input_ids(
        original_model=DecoderWrapper(
            original_model.model,
            original_model.lm_head,
        ),
        reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
        input_ids=input_ids,
        atol=1e-04,
    )
  except AssertionError as e:
    logging.error("*** FAILED *** verify with input IDs: %s", e)
  else:
    logging.info("*** PASSED *** verify with input IDs: %s", input_ids)


if __name__ == "__main__":
  app.run(main)
