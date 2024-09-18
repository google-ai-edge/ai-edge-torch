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

"""Verifies the reauthored Phi-2 model."""

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.utilities import verifier
import kagglehub
import transformers

_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)


def main(_):
  checkpoint = kagglehub.model_download("Microsoft/phi/transformers/2")
  verifier.log_msg("Loading the original model from", checkpoint)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(checkpoint)

  verifier.log_msg("Building the reauthored model from", checkpoint)
  reauthored_model = phi2.build_model(checkpoint)

  verifier.log_msg("Loading the tokenizer from", checkpoint)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

  verifier.verify_reauthored_model(
      original_model=original_model,
      reauthored_model=reauthored_model,
      tokenizer=tokenizer,
      prompts=_PROMPTS.value,
      atol=1e-03,
  )


if __name__ == "__main__":
  app.run(main)
