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
"""Utils for verifying the Qwen model."""

import logging
import os
import pathlib
from typing import Callable, Dict

from ai_edge_torch.generative.examples.qwen import qwen, qwen3
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import torch
import transformers


_BUILDER_V2 = {
    "0.5b": qwen.build_0_5b_model,
    "1.5b": qwen.build_1_5b_model,
    "3b": qwen.build_3b_model,
}

_BUILDER_V3 = {
    "0.6b": qwen3.build_0_6b_model,
    "1.7b": qwen3.build_1_7b_model,
    "4b": qwen3.build_4b_model,
}

_BUILDER = {
    "v2": _BUILDER_V2,
    "v3": _BUILDER_V3,
}

DEFAULT_PROMPTS = ["What is the meaning of life?"]


def verify_qwen(
    model_size: str,
    model_version: str,
    checkpoint_dir: str,
    weight_filename: str = "model.safetensors",
    max_new_tokens: int = 30,
    prompts: list[str] | None = None,
    initialize_from_local: bool = True,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
) -> bool:
  """Verifies the reauthored Llama 3.2 model with a custom loader."""
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
    reauthored_checkpoint = os.path.join(checkpoint_dir, weight_filename)

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = _BUILDER[model_version][model_size](
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
