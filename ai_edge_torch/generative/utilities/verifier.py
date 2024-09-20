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

"""Common utility functions to verify the reauthored models."""

import datetime
from typing import List, Optional, Union

from ai_edge_torch.generative.layers import kv_cache as kv_utils
import numpy as np
import torch
import transformers


def log_msg(*args):
  print("[%s]" % datetime.datetime.now(), *args)


class ModelWrapper(torch.nn.Module):
  """A wrapper for the model to be verified, this could be a HuggingFace model

  or a regular PyTorch model.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      model_format: str = "huggingface",
      hf_generation_config: Optional[transformers.GenerationConfig] = None,
  ):
    """Initializes the wrapper.

    Args:
      model (torch.nn.Module): The original model. This could be a model built
        from HuggingFace transformers, or a regular PyTorch model.
      model_format (str): The format of the model. It should be either
        "huggingface" or "pytorch".
      hf_generation_config (transformers.GenerationConfig): The HuggingFace
        generation config. This config will only be used if the underlying model
        is built from HuggingFace transformers.
    """
    super().__init__()
    self.model = model
    self.model_format = model_format
    self.hf_generation_config = hf_generation_config

  def generate(
      self, inputs: torch.Tensor
  ) -> Union[transformers.utils.ModelOutput, torch.LongTensor]:
    if self.model_format == "huggingface":
      return self.model.generate(
          inputs=inputs, generation_config=self.hf_generation_config
      )
    else:
      raise NotImplementedError(
          "generate() is not implemented for model format: %s"
          % self.model_format
      )

  def forward(
      self,
      inputs: torch.Tensor,
  ):
    return self.model.forward(inputs)


def forward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    kv_cache: kv_utils.KVCache,
) -> tuple[torch.Tensor, kv_utils.KVCache]:
  """Forwards the model reauthored with ai_edge_torch Generative API.

  Args:
    model (torch.nn.Module): The model to forward. It should be a model built
      with ai_edge_torch Generative API.
    tokens (torch.Tensor): The input tokens to forward.
    kv_cache (KVCache): The KV cache to forward.

  Returns:
    The output logits and the updated KV cache.
  """
  input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
  output = model.forward(tokens, input_pos, kv_cache)
  return output["logits"], output["kv_cache"]


def generate(
    model: torch.nn.Module, prompts: torch.Tensor, response_len: int
) -> torch.Tensor:
  """Generates the response to the prompts.

  It appends tokens output by the model to the prompts and feeds them back to
  the model up to decode_len.

  Args:
    model (torch.nn.Module): The model to generate. It should be a model built
      with ai_edge_torch Generative API.
    prompts (torch.Tensor): The prompts to generate.
    response_len (int): The number of tokens to generate.

  Returns:
    The generated tokens.
  """
  input_ids = prompts[0].int().tolist()
  kv_cache = kv_utils.KVCache.from_model_config(model.config)
  for _ in range(response_len - len(input_ids)):
    logits, kv_cache = forward(model, torch.tensor([input_ids]), kv_cache)
    generated_token = logits[0][-1].argmax().item()
    input_ids.append(generated_token)
  return torch.tensor([input_ids])


def verify_with_input_ids(
    original_model: ModelWrapper,
    reauthored_model: torch.nn.Module,
    input_ids: torch.Tensor = torch.from_numpy(np.array([[1, 2, 3, 4]])).int(),
    kv_cache_max_len: int = 1024,
    rtol: float = 1e-05,
    atol: float = 1e-05,
) -> bool:
  """Verifies if the model reauthored generates the same output of the oringal.

  It compares only one outputs from the original and the reauthored model.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (torch.nn.Module): The model reauthored with ai_edge_torch
      Generative API.
    input_ids (torch.Tensor): The input token IDs to forward.
    kv_cache_max_len (int): The maximum sequence length of the KV cache.
    rtol (float): The relative tolerance for the comparison.
    atol (float): The absolute tolerance for the comparison.

  Returns:
    True if the model reauthored generates the same output of the original.
  """
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.int, device="cpu")
  input_ids_len = input_ids.shape[1]
  tokens[0, :input_ids_len] = input_ids

  log_msg("Forwarding the original model...")
  outputs_original = original_model.forward(tokens)
  logits_original = outputs_original.logits[0, input_ids_len - 1, :]
  log_msg("logits_original: ", logits_original)

  log_msg("Forwarding the reauthored model...")
  kv_cache = kv_utils.KVCache.from_model_config(reauthored_model.config)
  outputs_reauthored = forward(reauthored_model, tokens, kv_cache)
  logits_reauthored = outputs_reauthored[0][0, input_ids_len - 1, :]
  log_msg("logits_reauthored:", logits_reauthored)

  return torch.allclose(
      logits_original, logits_reauthored, rtol=rtol, atol=atol
  )


def verify_model_with_prompts(
    original_model: ModelWrapper,
    reauthored_model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    prompts: str,
) -> bool:
  """Verifies if the model reauthored generates the same answer of the oringal.

  It compares an answer, i.e. multiple continuous outputs generated by the
  original and the reauthored model.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (torch.nn.Module): The model reauthored with ai_edge_torch
      Generative API.
    tokenizer (torch.nn.Module): The tokenizer.
    prompts (str): The input prompts to generate answers.

  Returns:
    True if the model reauthored generates the same answer of the original.
  """
  prompt_tokens = tokenizer.encode(prompts, return_tensors="pt")

  log_msg("Generating answer with the original model...")
  outputs_original = original_model.generate(prompt_tokens)
  response_original = tokenizer.decode(outputs_original[0])
  log_msg("outputs_from_original_model: [[", response_original, "]]")

  log_msg("Generating answer with the reauthored model...")
  generate_len = len(outputs_original[0])
  outputs_reauthored = generate(reauthored_model, prompt_tokens, generate_len)
  response_reauthored = tokenizer.decode(outputs_reauthored[0])
  log_msg("outputs from reauthored model: [[", response_reauthored, "]]")

  return response_original == response_reauthored


def verify_reauthored_model(
    original_model: ModelWrapper,
    reauthored_model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    prompts: List[str],
    rtol: float = 1e-05,
    atol: float = 1e-05,
):
  """Verifies the reauthored model against the original model.

  It verifies the reauthored model with two methods:
  1. It compares the output of the original and the reauthored model with an
     arbitrary input.
  2. It compares the answer generated by the original and the reauthored model
     with a prompt.

  It prints out "PASS" or "FAILED" to the console.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (torch.nn.Module): The model reauthored with ai_edge_torch
      Generative API.
    tokenizer (torch.nn.Module): The tokenizer.
    prompts (List[str]): List of the input prompts to generate answers.
    rtol (float): The relative tolerance for the comparison.
    atol (float): The absolute tolerance for the comparison.
  """
  log_msg("Verifying the reauthored model with an arbitrary input...")
  if verify_with_input_ids(
      original_model, reauthored_model, rtol=rtol, atol=atol
  ):
    log_msg("PASS")
  else:
    log_msg("FAILED")

  for p in prompts:
    log_msg("Verifying the reauthored model with prompts:", p)
    if verify_model_with_prompts(
        original_model, reauthored_model, tokenizer, p
    ):
      log_msg("PASS")
    else:
      log_msg("FAILED")
