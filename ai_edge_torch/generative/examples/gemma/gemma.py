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

"""Example of building a Gemma model."""

import os
import pathlib

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import numpy as np
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head=None,
)


class Gemma(nn.Module):
  """A Gemma model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim,
        config.vocab_size,
        bias=config.lm_head_use_bias,
    )
    # Gemma re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data
    # Gemma has only one block config.
    block_config = config.block_config(0)
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(block_config, config)
        for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    attn_config = block_config.attn_config
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=10_000,
        condense_ratio=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )
    assert len(self.transformer_blocks) == len(kv_cache.caches), (
        "The number of transformer blocks and the number of KV cache entries"
        " must be the same."
    )

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.kv_cache_max]

    # token embeddings of shape (b, t, n_embd)
    x = self.tok_embedding(tokens)
    x = x * (self.config.embedding_dim**0.5)

    updated_kv_entires = []
    for i, block in enumerate(self.transformer_blocks):
      kv_entry = kv_cache.caches[i] if kv_cache else None
      x, kv_entry = block(x, (cos, sin), mask, input_pos, kv_entry)
      if kv_entry:
        updated_kv_entires.append(kv_entry)
    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entires))

    x = self.final_norm(x)
    logits = self.lm_head(x)  # (b, t, vocab_size)
    return {"logits": logits, "kv_cache": updated_kv_cache}


def get_model_config_2b(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Gemma 2B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a Gemma 2B model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=8,
      head_dim=256,
      num_query_groups=1,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=16384,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=True,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=256000,
      num_layers=18,
      max_seq_len=8192,
      embedding_dim=2048,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_model_config_2b(kv_cache_max_len)
  # Gemma has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  return config


def build_2b_model(checkpoint_path: str, **kwargs) -> nn.Module:
  config = get_model_config_2b(**kwargs)
  model = Gemma(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # Since embedding and lm-head use the same weight, we need to set strict
  # to False.
  loader.load(model, strict=False)
  model.eval()
  return model


def define_and_run_2b(checkpoint_path: str) -> None:
  """Instantiates and runs a Gemma 2B model."""

  current_dir = pathlib.Path(__file__).parent.resolve()
  gemma_goldens = torch.load(current_dir / "gemma_lm_logits.pt")

  kv_cache_max_len = 1024
  model = build_2b_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.int, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len, dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(model.config)
  output = model.forward(tokens, input_pos, kv)
  print("comparing with goldens..")
  assert torch.allclose(
      gemma_goldens, output["logits"][0, idx.shape[1] - 1, :], atol=1e-02
  )


if __name__ == "__main__":
  input_checkpoint_path = os.path.join(
      pathlib.Path.home(), "Downloads/llm_data/gemma-2b"
  )
  define_and_run_2b(input_checkpoint_path)
