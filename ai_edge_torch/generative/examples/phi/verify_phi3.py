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

"""Verifies the reauthored Phi-3.5 model."""

import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.phi import phi3
from ai_edge_torch.generative.utilities import verifier
import transformers

_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "Instruct: Write an email about the weather Output:",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)


def main(_):
  checkpoint = "microsoft/Phi-3.5-mini-instruct"
  verifier.log_msg("Loading the original model from", checkpoint)
  generation_config = transformers.GenerationConfig.from_pretrained(checkpoint)
  generation_config.max_new_tokens = _MAX_NEW_TOKENS.value
  wrapper_model = verifier.ModelWrapper(
      model=transformers.AutoModelForCausalLM.from_pretrained(checkpoint),
      hf_generation_config=generation_config,
  )

  # Locate the cached dir.
  cached_config_file = transformers.utils.cached_file(
      checkpoint, transformers.utils.CONFIG_NAME
  )
  reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  verifier.log_msg("Building the reauthored model from", reauthored_checkpoint)
  reauthored_model = phi3.build_model(reauthored_checkpoint)

  verifier.log_msg("Loading the tokenizer from", checkpoint)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

  verifier.verify_reauthored_model(
      original_model=wrapper_model,
      reauthored_model=reauthored_model,
      tokenizer=tokenizer,
      generate_prompts=_PROMPTS.value,
  )


if __name__ == "__main__":
  app.run(main)
