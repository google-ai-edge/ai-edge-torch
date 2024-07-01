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
# Example of building a Gemma model.

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    pre_ff_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head=None,
)


class Gemma(nn.Module):

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    self.config = config
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
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(config) for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(config.attn_config.rotary_percentage * config.head_dim),
        base=10_000,
        condense_ratio=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max, dtype=torch.float32, device=torch.device("cpu")
    )
    self.config = config

  # The model's forward function takes in additional k/v cache tensors
  # and returns the updated k/v cache tensors to the caller.
  # This can be eliminated if we handle k/v cache updates inside the model itself.
  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    B, T = idx.size()
    assert (
        self.config.max_seq_len >= T
    ), f"Cannot forward sequence of length {T}, max seq length is only {self.config.max_seq_len}"

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.kv_cache_max]

    # token embeddings of shape (b, t, n_embd)
    x = self.tok_embedding(idx)
    x = x * (self.config.embedding_dim**0.5)

    for i, block in enumerate(self.transformer_blocks):
      x = block(x, (cos, sin), mask, input_pos)

    x = self.final_norm(x)
    res = self.lm_head(x)  # (b, t, vocab_size)
    return res


def get_model_config_2b(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  attn_config = cfg.AttentionConfig(
      num_heads=8,
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
  config = cfg.ModelConfig(
      vocab_size=256000,
      num_layers=18,
      max_seq_len=8192,
      embedding_dim=2048,
      kv_cache_max_len=kv_cache_max_len,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      pre_ff_norm_config=norm_config,
      final_norm_config=norm_config,
      parallel_residual=False,
      lm_head_use_bias=False,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config_2b_for_test() -> cfg.ModelConfig:
  config = get_model_config_2b()
  config.num_layers = 2
  return config


def build_2b_model(checkpoint_path, **kwargs) -> nn.Module:
  config = get_model_config_2b(**kwargs)
  model = Gemma(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # since embedding and lm-head use the same weight, we need to set strict
  # to False.
  loader.load(model, strict=False)
  return model


def define_and_run_2b() -> None:
  kv_cache_max_len = 1024
  checkpoint_path = os.path.join(Path.home(), "Downloads/llm_data/gemma-2b")
  model = build_2b_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.long, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len)
  print("running an inference")
  print(model.forward(tokens, input_pos))


if __name__ == "__main__":
  define_and_run_2b()
