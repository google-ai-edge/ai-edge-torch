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

"""Utility functions to verify the reauthored Gemma model."""

import logging
import os
from typing import Callable, Dict, List, Tuple

from ai_edge_torch.generative.examples.gemma import gemma1
from ai_edge_torch.generative.examples.gemma import gemma2
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.kv_cache as kv_utils
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import verifier
from gemma import config as gemma_config
from gemma import model as gemma_model
import torch


class GemmaWrapper(verifier.ModelWrapper):
  """Gemma model wrapper for verification.

  Verifier calls model.forward() with maxium sequence length (1024) expecting
  the output is logits while Gemma gets the input tokens with the actual length
  and returns logits in a tuple.

  Verifier runs tokenizer before model.generate() while Gemma runs the tokenizer
  inside model.generate().
  """

  def _get_actual_input_len(self, tokens: torch.Tensor) -> int:
    for i in range(tokens.shape[1]):
      if tokens[0, i] == 0:
        return i
    return tokens.shape[1]

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
    actual_input_len = self._get_actual_input_len(tokens)
    input_pos = torch.arange(0, actual_input_len, dtype=torch.long)
    mask_cache = attn_utils.build_causal_mask_cache(tokens.shape[1])
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
    weight_filename: str = "model.ckpt",
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
    tokenizer_filename: str = "tokenizer.model",
    max_new_tokens: int = 20,
    mask_as_input: bool = False,
    kv_layout: kv_utils.KVLayout = kv_utils.KV_LAYOUT_DEFAULT,
    rtol: float = 1e-05,
    atol: float = 1e-05,
) -> bool:
  """Verifies the reauthored Gemma model against the original model.

  Returns True if the verification passes, False otherwise.
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
      reauthored_model=verifier.ReauthoredModelWrapper(
          reauthored_model,
          mask_as_input=mask_as_input,
          kv_layout=kv_layout,
      ),
      tokenizer=GemmaTokenizerWrapper(original_model.tokenizer),
      generate_prompts=generate_prompts,
      max_new_tokens=max_new_tokens,
      forward_input_ids=forward_input_ids,
      rtol=rtol,
      atol=atol,
  )


def verify_gemma2(
    checkpoint_dir: str,
    weight_filename: str,
    prompts: List[str],
    max_new_tokens: int,
    mask_as_input: bool = False,
    kv_layout: kv_utils.KVLayout = kv_utils.KV_LAYOUT_DEFAULT,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
) -> bool:
  """Verifies the reauthored Gemma2 model.

  Return True if the verification passes, False otherwise.
  """
  checkpoint_path = os.path.join(checkpoint_dir, weight_filename)
  logging.info("Building the reauthored model from: %s", checkpoint_path)
  reauthored_model = gemma2.build_2b_model(checkpoint_path, custom_loader)

  return verify_reauthored_gemma_model(
      checkpoint=checkpoint_dir,
      variant="2b-v2",
      reauthored_model=reauthored_model,
      generate_prompts=prompts,
      forward_input_ids=[[2, 651, 9456, 576, 573, 3520, 3858, 603, 235248]],
      weight_filename=weight_filename,
      custom_loader=custom_loader,
      max_new_tokens=max_new_tokens,
      mask_as_input=mask_as_input,
      kv_layout=kv_layout,
      atol=1e-04,
  )


def verify_gemma1_with_custom_loader(checkpoint_dir: str) -> bool:
  """Verifies the reauthored Gemma1 model with a custom loader."""
  weight_filename = "gemma-2b-it.ckpt"
  checkpoint_path = os.path.join(checkpoint_dir, weight_filename)
  custom_loader = loader.get_custom_loader(checkpoint_path)
  reauthored_model = gemma1.build_2b_model(checkpoint_path, custom_loader)
  return verify_reauthored_gemma_model(
      checkpoint=checkpoint_dir,
      variant="2b",
      reauthored_model=reauthored_model,
      weight_filename=weight_filename,
      custom_loader=custom_loader,
      generate_prompts=["What is the meaning of life?"],
      forward_input_ids=[[1, 2, 3, 4]],
      max_new_tokens=30,
  )


def verify_gemma2_with_custom_loader(checkpoint_dir: str) -> bool:
  """Verifies the reauthored Gemma2 model with a custom loader."""
  return verify_gemma2(
      checkpoint_dir=checkpoint_dir,
      weight_filename="model.ckpt",
      prompts=["What is the meaning of life?"],
      max_new_tokens=30,
      mask_as_input=True,
      custom_loader=loader.get_custom_loader("", checkpoint_format="pt"),
  )
