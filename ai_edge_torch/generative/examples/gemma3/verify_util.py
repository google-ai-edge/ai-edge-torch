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

"""Utility functions to verify the reauthored Gemma model."""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import verifier
from gemma import config as gemma_config
from gemma import model as gemma_model
import torch


def _get_actual_input_len(tokens: torch.Tensor) -> int:
  for i in range(tokens.shape[1]):
    if tokens[0, i] == 0:
      return i
  return tokens.shape[1]


class GemmaWrapper(verifier.ModelWrapper):
  """Gemma model wrapper for verification.

  Verifier calls model.forward() with maxium sequence length (1024) expecting
  the output is logits while Gemma gets the input tokens with the actual length
  and returns logits in a tuple.

  Verifier runs tokenizer before model.generate() while Gemma runs the tokenizer
  inside model.generate().
  """

  def _get_kv_caches(
      self, max_seq_len: int
  ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    config = self.model.config
    cache_size = (1, max_seq_len, config.num_key_value_heads, config.head_dim)
    cache = torch.zeros(cache_size)
    return [
        (cache.clone(), cache.clone()) for _ in range(config.num_hidden_layers)
    ]

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    """Forwards the model after reducing input tokens to the actual length."""
    actual_input_len = _get_actual_input_len(tokens)
    input_pos = torch.arange(0, actual_input_len, dtype=torch.long)
    mask_cache = attn_utils.build_causal_mask_cache(tokens.shape[1])
    local_mask_cache = attn_utils.build_sliding_window_mask_cache(
        tokens.shape[1], self.model.config.sliding_window_size
    )
    _, logits = self.model.forward(
        input_token_ids=tokens[0, :actual_input_len].unsqueeze(0),
        input_positions=input_pos,
        kv_write_indices=None,
        kv_caches=self._get_kv_caches(tokens.shape[1]),
        mask=mask_cache.index_select(2, input_pos),
        output_positions=input_pos,
        temperatures=None,
        top_ps=torch.tensor([1.0], dtype=torch.float),
        top_ks=torch.tensor([1], dtype=torch.long),
        local_mask=local_mask_cache.index_select(2, input_pos),
    )
    return logits

  def generate(
      self, tokens: torch.Tensor, max_new_tokens: int
  ) -> torch.IntTensor:
    """Generates the response after decoding the tokens into a string."""
    prompts = self.model.tokenizer.decode(tokens[0].tolist())
    response = self.model.generate(
        prompts, device="cpu", output_len=max_new_tokens, top_k=1
    )
    return torch.tensor([self.model.tokenizer.encode(prompts + response)])


class UnifiedGemma3Wrapper(verifier.ReauthoredModelWrapper):
  """Unified Gemma3 model wrapper for verification."""

  def __init__(self, model: torch.nn.Module):
    super().__init__(model, kv_layout=kv_utils.KV_LAYOUT_TRANSPOSED)

  def _init_kv_cache(self):
    return kv_utils.KVCache.from_model_config(
        self.model.model.config, kv_layout=self.kv_layout
    )

  def forward(
      self, tokens: torch.Tensor, pixel_values: torch.Tensor = None
  ) -> torch.Tensor:
    """Forwards the model."""
    mask = attn_utils.build_causal_mask_cache(
        self.model.model.config.kv_cache_max_len
    )
    input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
    mask = mask.index_select(2, input_pos)
    output = self.model.model.forward(
        tokens, input_pos, self._init_kv_cache(), mask=mask
    )
    return output["logits"]

  def generate(
      self,
      prompts: torch.Tensor,
      max_new_tokens: int,
      pixel_values: torch.Tensor = None,
      eos_token_id: Optional[int] = None,
  ) -> torch.IntTensor:
    """Generates the response."""
    input_ids = prompts[0].int().tolist()
    tokens = torch.tensor([input_ids])
    input_pos = torch.arange(0, tokens.shape[1], dtype=torch.int)
    kv_cache = self._init_kv_cache()
    mask_cache = attn_utils.build_causal_mask_cache(
        self.model.model.config.kv_cache_max_len
    )
    for _ in range(max_new_tokens):
      mask = mask_cache.index_select(2, input_pos)
      output = self.model.model.forward(
          tokens, input_pos, kv_cache, mask=mask
      )
      logits, kv_cache = output["logits"], output["kv_cache"]
      generated_token = logits[0][-1].argmax().item()
      input_ids.append(generated_token)
      if eos_token_id is not None and generated_token == eos_token_id:
        break
      tokens = torch.tensor([[generated_token]])
      input_pos = torch.tensor([len(input_ids) - 1])
    return torch.tensor([input_ids])


class GemmaTokenizerWrapper(verifier.TokenizerWrapper):
  """Tokenizer wrapper for verification.

  Verifier expects the tokenizer to handle tokens in torch.Tensor while Gemma
  tokenizer expects tokens in a list.
  """

  def encode(self, text: str, **_) -> torch.Tensor:
    """Adds one more dimension to the output of the tokenizer."""
    return torch.tensor([self.tokenizer.encode(text)])

  def decode(self, tokens: torch.Tensor) -> str:
    """Decodes the token sequence after converting to a list."""
    return self.tokenizer.decode(tokens.tolist())


def verify_reauthored_gemma_model(
    checkpoint: str,
    variant: str,
    reauthored_model: torch.nn.Module,
    generate_prompts: List[str],
    forward_input_ids: List[List[int]],
    weight_filename: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    tokenizer_filename: str = "tokenizer.model",
    max_new_tokens: int = 20,
    rtol: float = 1e-05,
    atol: float = 1e-05,
) -> bool:
  """Verifies the reauthored Gemma model against the original model.

  Args:
      checkpoint: Path to the Gemma checkpoint.
      variant: Gemma model variant.
      reauthored_model: The reauthored model to verify.
      generate_prompts: List of prompts for generation.
      forward_input_ids: List of input ids for forward pass.
      weight_filename: Name of the weight file.
      tokenizer_filename: Name of the tokenizer file.
      max_new_tokens: Maximum number of new tokens to generate.
      rtol: Relative tolerance for comparison.
      atol: Absolute tolerance for comparison.

  Returns:
      True if the verification passes, False otherwise.
  """
  config = gemma_config.get_model_config(variant)
  config.tokenizer = os.path.join(checkpoint, tokenizer_filename)
  # Use float32 to be compatible with the reauthored model.
  config.dtype = torch.float32

  logging.info("Loading the original model from: %s", checkpoint)
  original_model = gemma_model.GemmaForCausalLM(config).eval()
  checkpoint_path = os.path.join(checkpoint, weight_filename)
  if custom_loader is None:
    original_model.load_weights(checkpoint_path)
  else:
    original_model.load_state_dict(
        custom_loader(checkpoint_path)["model_state_dict"],
        strict=False,
    )

  return verifier.verify_reauthored_model(
      original_model=GemmaWrapper(original_model),
      reauthored_model=UnifiedGemma3Wrapper(reauthored_model),
      tokenizer=GemmaTokenizerWrapper(original_model.tokenizer),
      generate_prompts=generate_prompts,
      max_new_tokens=max_new_tokens,
      forward_input_ids=forward_input_ids,
      rtol=rtol,
      atol=atol,
  )


def verify_gemma3(
    checkpoint: str,
    prompts: List[str],
    max_new_tokens: int,
    variant: str,
    weight_filename: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
) -> bool:
  """Verifies the reauthored Gemma3 model.

  Args:
      checkpoint: Path to the Gemma checkpoint.
      prompts: List of prompts for generation.
      max_new_tokens: Maximum number of new tokens to generate.
      variant: Gemma model variant.
      weight_filename: Name of the weight file.
      custom_loader: A custom loader to load the weights.

  Returns:
      True if the verification passes, False otherwise.
  """
  gemma3_model_path = os.path.join(checkpoint, weight_filename)
  logging.info("Building the reauthored model from: %s", gemma3_model_path)

  if variant == "1b":
    reauthored_model = UnifiedGemma3Wrapper(
        gemma3.build_model_1b(gemma3_model_path, custom_loader)
    )
  else:
    raise ValueError(f"Unsupported Gemma3 variant: {variant}")

  return verify_reauthored_gemma_model(
      checkpoint=checkpoint,
      variant=variant,
      reauthored_model=reauthored_model,
      generate_prompts=prompts,
      forward_input_ids=[[2, 651, 9456, 576, 573, 3520, 3858, 603, 235248]],
      max_new_tokens=max_new_tokens,
      weight_filename=weight_filename,
      custom_loader=custom_loader,
      atol=1e-04,
  )


def verify_gemma3_with_custom_loader(checkpoint: str) -> bool:
  """Verifies the reauthored Gemma3 model with a custom loader."""
  return verify_gemma3(
      checkpoint=checkpoint,
      prompts=["What is the meaning of life?"],
      max_new_tokens=30,
      variant="1b",
      weight_filename="model.ckpt",
      custom_loader=loader.get_custom_loader("", checkpoint_format="pt"),
  )
