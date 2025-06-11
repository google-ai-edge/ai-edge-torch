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
"""Utils for verifying the Hammer 2.1 model."""
import logging
import os
import pathlib

from ai_edge_torch.generative.examples.hammer import hammer
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import transformers


_BUILDER = {
    "0.5b": hammer.build_0_5b_model,
    "1.5b": hammer.build_1_5b_model,
}

DEFAULT_PROMPTS = ["What is the meaning of life?"]


def verify_hammer(
    model_size: str,
    checkpoint_dir: str,
    weight_filename: str = "model.safetensors",
    max_new_tokens: int = 30,
    initialize_from_local: bool = True,
    prompts: list[str] | None = None,
) -> bool:
  """Verifies the reauthored Hammer 2.1 model with a custom loader."""
  logging.info("Loading the original model from: %s", checkpoint_dir)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(
      checkpoint_dir
  )

  logging.info("Building the reauthored model from: %s", checkpoint_dir)
  custom_loader = (
      None
      if initialize_from_local
      else loader.get_custom_loader("", "safetensors")
  )

  if initialize_from_local:
    # Locate the cached dir.
    cached_config_file = transformers.utils.cached_file(
        checkpoint_dir, transformers.utils.CONFIG_NAME
    )
    reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  else:
    reauthored_checkpoint = os.path.join(checkpoint_dir, weight_filename)

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = _BUILDER[model_size](
      checkpoint_path=reauthored_checkpoint,
      custom_loader=custom_loader,
      mask_cache_size=verifier.DEFAULT_KV_CACHE_MAX_LEN,
  )

  logging.info("Loading the tokenizer from: %s", checkpoint_dir)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
  return verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=DEFAULT_PROMPTS if prompts is None else prompts,
      max_new_tokens=max_new_tokens,
      atol=1e-04,
  )
