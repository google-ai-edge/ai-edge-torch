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
# Example of building a TinyLlama model from the Edge Generative API layers.

import os
from pathlib import Path

from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import numpy as np
import torch
import torch.nn as nn

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
    lm_head="lm_head",
)


class TinyLLamma(nn.Module):

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    self.config = config
    # Construct model layers.
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(config) for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(
            config.attn_config.rotary_percentage * config.attn_config.head_dim
        ),
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

  # The model's forward function takes in additional k/v cache tensors
  # and returns the updated k/v cache tensors to the caller.
  # This can be eliminated if we handle k/v cache updates inside the model itself.
  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    B, T = idx.size()
    assert self.config.max_seq_len >= T, (
        f"Cannot forward sequence of length {T}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.kv_cache_max]

    # forward the model itself
    x = self.tok_embedding(idx)  # token embeddings of shape (b, t, n_embd)

    for i, block in enumerate(self.transformer_blocks):
      x = block(x, (cos, sin), mask, input_pos)

    x = self.final_norm(x)

    res = self.lm_head(x)  # (b, t, vocab_size)
    return res


def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=64,
      num_query_groups=4,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=5632,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  config = cfg.ModelConfig(
      vocab_size=32000,
      num_layers=22,
      max_seq_len=2048,
      embedding_dim=2048,
      kv_cache_max_len=kv_cache_max_len,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      pre_ff_norm_config=norm_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config_for_test() -> cfg.ModelConfig:
  config = get_model_config()
  config.vocab_size = 128
  config.num_layers = 2
  config.ff_config.intermediate_size = 256
  return config


def build_model(checkpoint_path, **kwargs) -> nn.Module:
  config = get_model_config(**kwargs)
  model = TinyLLamma(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(model)
  return model


def define_and_run() -> None:
  current_dir = Path(__file__).parent.resolve()
  tiny_llama_goldens = torch.load(current_dir / "tiny_llama_lm_logits.pt")
  kv_cache_max_len = 1024
  checkpoint_path = os.path.join(Path.home(), "Downloads/llm_data/tiny_llama")
  model = build_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.long, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len)
  lm_logits = model.forward(tokens, input_pos)
  assert torch.allclose(
      tiny_llama_goldens, lm_logits[0, idx.shape[1] - 1, :], atol=1e-05
  )


if __name__ == "__main__":
  define_and_run()
