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

"""Verifies the reauthored Gemma2 model."""

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.gemma import verify_util
import ai_edge_torch.generative.layers.kv_cache as kv_utils
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
_MASK_AS_INPUT = flags.DEFINE_bool(
    "mask_as_input",
    True,
    "Pass the causal self attention mask to the model.",
)
_TRANSPOSE_KV_CACHE = flags.DEFINE_bool(
    "transpose_kv_cache",
    True,
    "Transpose the KV cache to reduce memory usage.",
)
_WEIGHT_FILENAME = flags.DEFINE_string(
    "weight_filename",
    "model.ckpt",
    "Name of the weight file in the checkpoint directory.",
)

def main(_):
  checkpoint = kagglehub.model_download("google/gemma-2/pyTorch/gemma-2-2b-it")

  verify_util.verify_gemma2(
      checkpoint,
      _WEIGHT_FILENAME.value,
      _PROMPTS.value,
      _MAX_NEW_TOKENS.value,
      _MASK_AS_INPUT.value,
      kv_utils.KV_LAYOUT_TRANSPOSED if _TRANSPOSE_KV_CACHE.value else kv_utils.KV_LAYOUT_DEFAULT,
  )


if __name__ == "__main__":
  app.run(main)
