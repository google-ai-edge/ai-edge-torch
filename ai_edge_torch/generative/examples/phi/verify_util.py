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
"""Utils for verifying the Phi model."""

import logging
import pathlib
from typing import Callable, Dict

from ai_edge_torch.generative.examples.phi import phi2, phi3, phi4
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import torch
import transformers


DEFAULT_PROMPTS = ["Instruct: Write an email about the weather Output:"]

_BUILDER = {
    "v2": phi2.build_model,
    "v3": phi3.build_model,
    "v4": phi4.build_model,
}


def verify_phi(
    version: str,
    checkpoint_dir: str,
    max_new_tokens: int = 30,
    prompts: list[str] | None = None,
    atol: float = 1e-04,
    initialize_from_local: bool = True,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
) -> bool:
  """Verifies the reauthored Phi model with a custom loader."""
  logging.info("Loading the original model from: %s", checkpoint_dir)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(
      checkpoint_dir
  )

  logging.info("Building the reauthored model from: %s", checkpoint_dir)
  if custom_loader is None and not initialize_from_local:
    custom_loader = loader.get_custom_loader("", "safetensors")

  if initialize_from_local:
    # Locate the cached dir.
    cached_config_file = transformers.utils.cached_file(
        checkpoint_dir, transformers.utils.CONFIG_NAME
    )
    reauthored_checkpoint = pathlib.Path(cached_config_file).parent
  else:
    reauthored_checkpoint = checkpoint_dir

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = _BUILDER[version](
      checkpoint_path=reauthored_checkpoint,
      custom_loader=custom_loader,
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
      atol=atol,
  )
