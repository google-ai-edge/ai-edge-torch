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

"""Verifies the reauthored Qwen 2.5 0.5B, 1.5B, and 3B models."""


from absl import app
from absl import flags
from ai_edge_torch.generative.examples.qwen import verify_util


_MODEL_SIZE = flags.DEFINE_enum(
    "model_size",
    "3b",
    ["0.5b", "1.5b", "3b"],
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
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}


def main(_):
  verify_util.verify_qwen(
      model_size=_MODEL_SIZE.value,
      model_version="v2",
      checkpoint_dir=_CHECKPOINT[_MODEL_SIZE.value],
      max_new_tokens=_MAX_NEW_TOKENS.value,
      prompts=_PROMPTS.value,
  )


if __name__ == "__main__":
  app.run(main)
