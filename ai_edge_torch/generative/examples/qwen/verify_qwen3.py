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

"""Verifies the reauthored Qwen 3.0 0.6B, 1.7B, and 4B models."""


from absl import app
from absl import flags
from ai_edge_torch.generative.examples.qwen import verify_util


_MODEL_SIZE = flags.DEFINE_enum(
    "model_size",
    "0.6b",
    ["0.6b", "1.7b", "4b"],
    "The size of the model to verify.",
)
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

_CHECKPOINT = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
}


def main(_):
  verify_util.verify_qwen(
      model_size=_MODEL_SIZE.value,
      model_version="v3",
      checkpoint_dir=_CHECKPOINT[_MODEL_SIZE.value],
      max_new_tokens=_MAX_NEW_TOKENS.value,
      prompts=_PROMPTS.value,
  )


if __name__ == "__main__":
  app.run(main)
