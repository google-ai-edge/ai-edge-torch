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

"""Verifies the reauthored TinyLlama-1.1B model."""


from absl import app
from absl import flags
from ai_edge_torch.generative.examples.tiny_llama import verify_util


_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "Show me the program to add 2 and 3.",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


def main(_):
  verify_util.verify_tiny_llama(
      checkpoint_dir="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      max_new_tokens=_MAX_NEW_TOKENS.value,
      prompts=_PROMPTS.value,
  )


if __name__ == "__main__":
  app.run(main)
