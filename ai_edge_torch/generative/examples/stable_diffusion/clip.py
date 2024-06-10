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

import torch
from torch import nn

from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attention_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="layers.{}.linear_1",
    ff_down_proj="layers.{}.linear_2",
    ff_gate_proj="layers.{}.linear_1",
    attn_fused_qkv_proj="layers.{}.attention.in_proj",
    attn_output_proj="layers.{}.attention.out_proj",
    pre_attn_norm="layers.{}.layernorm_1",
    pre_ff_norm="layers.{}.layernorm_2",
    embedding="embedding.token_embedding",
    embedding_position="embedding.position_value",
    final_norm="layernorm",
    lm_head=None,
)


class CLIP(nn.Module):
  """CLIP text encoder
  For details, see https://arxiv.org/abs/2103.00020
  """

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
    self.tok_embedding_position = nn.Parameter(
        torch.zeros((config.max_seq_len, config.embedding_dim))
    )

    self.config = config
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(config) for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(config.embedding_dim, config.final_norm_config)

    self.mask_cache = attention_utils.build_causal_mask_cache(
        size=config.max_seq_len, dtype=torch.float32
    )

  @torch.inference_mode
  def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
    tokens = tokens.type(torch.long)

    state = self.tok_embedding(tokens) + self.tok_embedding_position
    for layer in self.transformer_blocks:
      state = layer(state, mask=self.mask_cache)
    output = self.final_norm(state)
    return output


def get_model_config() -> cfg.ModelConfig:
  max_seq_len = 77
  vocab_size = 49408
  num_layers = 12
  num_heads = 12
  num_query_groups = 12
  embedding_dim = 768

  attn_config = cfg.AttentionConfig(
      num_heads=num_heads,
      num_query_groups=num_query_groups,
      rotary_percentage=0.0,
      qkv_use_bias=True,
      qkv_transpose_before_split=True,
      output_proj_use_bias=True,
      enable_kv_cache=False,
  )

  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_QUICK),
      intermediate_size=embedding_dim * 4,
      use_bias=True,
  )

  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.LAYER_NORM)

  config = cfg.ModelConfig(
      vocab_size=vocab_size,
      num_layers=num_layers,
      max_seq_len=max_seq_len,
      embedding_dim=embedding_dim,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      pre_ff_norm_config=norm_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )

  return config
