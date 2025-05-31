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

import logging
from typing import Any, List, Optional

from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.utilities import export_config
import torch

ExportConfig = export_config.ExportConfig


class ModelWrapper(torch.nn.Module):
  """A wrapper for the model to be verified.

  It unifies the interface of forward() and generate() of models for the
  verification to call.
  """

  def __init__(self, model: torch.nn.Module):
    """Initializes the wrapper.

    Args:
      model (torch.nn.Module): The model which might have different interfaces
        of forward() and generate(). It could be a model built from HuggingFace
        transformers, a regular PyTorch model, or a model re-authored with
        ai_edge_torch Generative API.
    """
    super().__init__()
    self.model = model
    self.export_config = ExportConfig(output_logits_on_prefill=True)

  def forward(
      self, tokens: torch.Tensor, pixel_values: torch.Tensor = None
  ) -> torch.Tensor:
    """Gets output logits by forwarding the input tokens.

    Args:
      tokens (torch.Tensor): The input tokens to forward. Its dimension is
        expected to be (batch_size=1, kv_cache_max_len).
      pixel_values (torch.Tensor): Optional input pixel values to forward.

    Returns:
      The output logits.
    """
    raise NotImplementedError("forward() is not implemented.")

  def generate(
      self,
      prompts: torch.Tensor,
      max_new_tokens: int,
      pixel_values: torch.Tensor = None,
  ) -> torch.IntTensor:
    """Returns the response token IDs to the given prompts tensor.

    The maximum number of tokens to generate might be set by subclasses.

    Args:
      prompts (torch.Tensor): The input token IDs to generate with. Its shape is
        expected to be (batch_size=1, input_ids_len).
      max_new_tokens (int): The maximum number of response token IDs to
        generate.
      pixel_values (torch.Tensor): Optional input pixel values to generate with.

    Returns:
      The tensor of response token IDs with shape of (batch_size=1,
      response_ids_len).
    """
    raise NotImplementedError("generate() is not implemented.")


class ReauthoredModelWrapper(ModelWrapper):
  """A wrapper for the model reauthored with ai_edge_torch Generative API."""

  def __init__(
      self,
      model: torch.nn.Module,
      mask_as_input: bool = False,
      kv_layout: kv_utils.KVLayout = kv_utils.KV_LAYOUT_DEFAULT,
      kv_cache_max_len: int = 1280,
  ):
    """Wraps a reauthored model with some options."""
    super().__init__(model)
    self.mask_as_input = mask_as_input
    self.kv_layout = kv_layout
    self.kv_cache_max_len = kv_cache_max_len

  def _init_kv_cache(self):
    """Returns an initialized KV cache."""
    return kv_utils.KVCache.from_model_config(
        self.kv_cache_max_len, self.model.config, kv_layout=self.kv_layout
    )

  def _get_extra_args_for_forward(self) -> dict[str, Any]:
    """Returns extra arguments for the forward() method."""
    return {}

  def _build_mask(self, input_pos: torch.Tensor) -> torch.Tensor:
    """Builds a mask for the model."""
    mask = torch.full(
        (len(input_pos), self.kv_cache_max_len),
        float("-inf"),
        dtype=torch.float32,
    )
    return torch.triu(mask, diagonal=input_pos[0] + 1).unsqueeze(0).unsqueeze(0)

  def _forward_with_kv_cache(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      pixel_values: torch.Tensor,
  ) -> tuple[torch.Tensor, kv_utils.KVCache]:
    """Forwards the model and updates an external KV cache.

    Args:
      tokens (torch.Tensor): The input tokens to forward.
      input_pos (torch.Tensor): The input positions to forward.
      kv_cache (KVCache): The KV cache to forward.
      pixel_values (torch.Tensor): The input pixel values to forward.

    Returns:
      The output logits and the updated KV cache.
    """
    extra_args = self._get_extra_args_for_forward()
    if self.export_config is not None:
      # Verification requires logit outputs on prefill for comparison.
      if not self.export_config.output_logits_on_prefill:
        raise ValueError("Verifier requires logit output on prefill.")
      extra_args["export_config"] = self.export_config
    if pixel_values is not None:
      extra_args["pixel_values"] = pixel_values
    if self.mask_as_input:
      extra_args["mask"] = self._build_mask(input_pos)
    output = self.model.forward(tokens, input_pos, kv_cache, **extra_args)
    return output["logits"], output["kv_cache"]

  def forward(
      self, tokens: torch.Tensor, pixel_values: torch.Tensor = None
  ) -> torch.Tensor:
    input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
    logits, _ = self._forward_with_kv_cache(
        tokens, input_pos, self._init_kv_cache(), pixel_values
    )
    return logits

  def generate(
      self,
      prompts: torch.Tensor,
      max_new_tokens: int,
      pixel_values: torch.Tensor = None,
      eos_token_id: Optional[int] = None,
  ) -> torch.IntTensor:
    input_ids = prompts[0].int().tolist()
    tokens = torch.tensor([input_ids])
    input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
    kv_cache = self._init_kv_cache()
    for _ in range(max_new_tokens):
      logits, kv_cache = self._forward_with_kv_cache(
          tokens, input_pos, kv_cache, pixel_values
      )
      generated_token = logits[0][-1].argmax().item()
      input_ids.append(generated_token)
      if eos_token_id is not None and generated_token == eos_token_id:
        break
      tokens = torch.tensor([[generated_token]])
      input_pos = torch.tensor([len(input_ids) - 1])
      pixel_values = None  # Pass only for the first time.
    return torch.tensor([input_ids])


class TokenizerWrapper(torch.nn.Module):
  """A wrapper for the tokenizer used for verification."""

  def __init__(self, tokenizer: torch.nn.Module):
    """Initializes the wrapper.

    Args:
      tokenizer (torch.nn.Module): The tokenizer to wrap.
    """
    super().__init__()
    self.tokenizer = tokenizer

  def encode(self, prompts: str) -> torch.Tensor:
    """Encodes the prompts to token IDs."""
    return self.tokenizer.encode(prompts, return_tensors="pt")

  def decode(self, token_ids: torch.Tensor) -> str:
    """Decodes the token IDs to a string."""
    return self.tokenizer.decode(token_ids)


def verify_with_input_ids(
    original_model: ModelWrapper,
    reauthored_model: ReauthoredModelWrapper,
    input_ids: List[int],
    total_seq_len: int = 128,
    rtol: float = 1e-05,
    atol: float = 1e-05,
):
  """Verifies if the model reauthored generates the same output of the oringal.

  It compares only one outputs from the original and the reauthored model.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (ReauthoredModelWrapper): The model reauthored with
      ai_edge_torch Generative API.
    input_ids (List[int]): The input token IDs to forward with.
    total_seq_len (int): The total sequence length of the input.
    rtol (float): The relative tolerance for the comparison.
    atol (float): The absolute tolerance for the comparison.

  Raises:
    AssertError if the model reauthored fails to generate the same output of the
      original.
  """
  tokens = torch.full((1, total_seq_len), 0, dtype=torch.int, device="cpu")
  tokens[0, : len(input_ids)] = torch.tensor([input_ids]).int()

  logging.info("Forwarding the original model...")
  outputs_original = original_model.forward(tokens)
  logits_original = outputs_original[0, len(input_ids) - 1, :]
  logging.info("logits_original: %s", logits_original)

  logging.info("Forwarding the reauthored model...")
  outputs_reauthored = reauthored_model.forward(tokens)
  logits_reauthored = outputs_reauthored[0, len(input_ids) - 1, :]
  logging.info("logits_reauthored: %s", logits_reauthored)

  assert torch.allclose(
      logits_original, logits_reauthored, rtol=rtol, atol=atol
  )


def verify_model_with_prompts(
    original_model: ModelWrapper,
    reauthored_model: ReauthoredModelWrapper,
    tokenizer: TokenizerWrapper,
    prompts: str,
    max_new_tokens: int,
):
  """Verifies if the model reauthored generates the same answer of the oringal.

  It compares an answer, i.e. multiple continuous outputs generated by the
  original and the reauthored model.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (ReauthoredModelWrapper): The model reauthored with
      ai_edge_torch Generative API.
    tokenizer (TokenizerWrapper): The tokenizer.
    prompts (str): The input prompts to generate answers.
    max_new_tokens (int): The maximum number of new tokens to generate.

  Raises:
    AssertError if the model reauthored fails to generate the same answer of the
      original.
  """
  prompt_tokens = tokenizer.encode(prompts)

  logging.info("Generating answer with the original model...")
  outputs_original = original_model.generate(prompt_tokens, max_new_tokens)
  response_original = tokenizer.decode(outputs_original[0])
  logging.info("outputs_from_original_model: [[%s]]", response_original)

  logging.info("Generating answer with the reauthored model...")
  outputs_reauthored = reauthored_model.generate(
      prompt_tokens,
      max_new_tokens,
      eos_token_id=getattr(tokenizer.tokenizer, "eos_token_id", None),
  )
  response_reauthored = tokenizer.decode(outputs_reauthored[0])
  logging.info("outputs from reauthored model: [[%s]]", response_reauthored)

  assert response_original == response_reauthored


def verify_reauthored_model(
    original_model: ModelWrapper,
    reauthored_model: ReauthoredModelWrapper,
    tokenizer: TokenizerWrapper,
    generate_prompts: List[str],
    max_new_tokens: int = 30,
    forward_input_ids: List[List[int]] = [[1, 2, 3, 4]],
    rtol: float = 1e-05,
    atol: float = 1e-05,
    continue_on_failure: bool = False,
    verify_inputs: bool = True,
    verify_prompts: bool = True,
) -> bool:
  """Verifies the reauthored model against the original model.

  It verifies the reauthored model with two methods:
  1. It compares the output of the original and the reauthored model with an
     arbitrary input.
  2. It compares the answer generated by the original and the reauthored model
     with a prompt.

  It prints out "PASS" or "FAILED" to the console. It returns True if all
  verification passes, False otherwise.

  Args:
    original_model (ModelWrapper): The original model.
    reauthored_model (ReauthoredModelWrapper): The model reauthored with
      ai_edge_torch Generative API.
    tokenizer (TokenizerWrapper): The tokenizer.
    generate_prompts (List[str]): List of the input prompts to generate answers.
    max_new_tokens (int): The maximum number of new tokens to generate.
    forward_input_ids (List[torch.Tensor]): List if ihe input token IDs to
      forward with.
    rtol (float): The relative tolerance for the comparison.
    atol (float): The absolute tolerance for the comparison.
    continue_on_failure (bool): If True, it continues to verify the next prompt
      or input IDs even if a previous one fails.
    verify_inputs (bool): If True, it verifies the model with forward_input_ids.
    verify_prompts (bool): If True, it verifies the model with generate_prompts.

  Returns:
    True if all verification passes, False otherwise.
  """
  failure_count = 0

  if verify_inputs:
    for input_ids in forward_input_ids:
      logging.info(
          "Verifying the reauthored model with input IDs: %s", input_ids
      )
      try:
        verify_with_input_ids(
            original_model, reauthored_model, input_ids, rtol=rtol, atol=atol
        )
      except AssertionError as e:
        logging.error("*** FAILED *** verify with input IDs: %s", input_ids)
        failure_count += 1
        if not continue_on_failure:
          return False
      else:
        logging.info("*** PASSED *** verify with input IDs: %s", input_ids)

  if verify_prompts:
    for prompts in generate_prompts:
      logging.info("Verifying the reauthored model with prompts: %s", prompts)
      try:
        verify_model_with_prompts(
            original_model, reauthored_model, tokenizer, prompts, max_new_tokens
        )
      except AssertionError as e:
        logging.error("*** FAILED *** verify with prompts: %s", prompts)
        failure_count += 1
        if not continue_on_failure:
          return False
      else:
        logging.info("*** PASSED *** verify with prompts: %s", prompts)

  if failure_count == 0:
    logging.info("*** PASSED *** verify_reauthored_model")
    return True
  else:
    logging.error("*** FAILED *** verify_reauthored_model")
    return False
