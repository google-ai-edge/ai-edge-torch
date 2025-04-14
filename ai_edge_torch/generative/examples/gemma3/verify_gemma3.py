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

"""Verifies the reauthored Gemma3 model."""

import glob
import logging
import os
from absl import app
from absl import flags
from ai_edge_torch.generative.examples.gemma3 import verify_util
import kagglehub


_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    "",
    "The checkpoint to verify.",
)
_VARIANT = flags.DEFINE_string(
    "variant",
    "1b",
    "The variant of the model to verify.",
)
_WEIGHT_FILENAME = flags.DEFINE_string(
    "weight_filename",
    None,
    "The weightfilename of the model to verify.",
)


def find_first_ckpt(folder):
  """Finds the first .ckpt file in a folder."""
  ckpt_files = sorted(glob.glob(os.path.join(folder, "*.ckpt")))
  return os.path.basename(ckpt_files[0]) if ckpt_files else None


def main(_):
  if _CHECKPOINT.value:
    checkpoint = _CHECKPOINT.value
  else:
    checkpoint = kagglehub.model_download(
        "google/gemma-3/pyTorch/gemma-3-1b-it"
    )

  # If the weight filename is not specified, use the first checkpoint.
  if _WEIGHT_FILENAME.value is None:
    weight_filename = find_first_ckpt(checkpoint)
    logging.info(
        "NOTE: using the first weight file `%s` from `%s`",
        weight_filename,
        checkpoint,
    )
  else:
    weight_filename = _WEIGHT_FILENAME.value

  # Verify the reauthored model by comparing the outputs with the original one.
  verify_util.verify_gemma3(
      checkpoint,
      _PROMPTS.value,
      _MAX_NEW_TOKENS.value,
      _VARIANT.value,
      weight_filename,
  )


if __name__ == "__main__":
  app.run(main)
