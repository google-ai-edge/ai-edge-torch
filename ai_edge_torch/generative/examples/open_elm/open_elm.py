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

import os
from pathlib import Path

import copy
import numpy as np
import torch
import torch.nn as nn

import ai_edge_torch.generative.examples.open_elm.loader as loading_utils
from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    embedding="transformer.token_embeddings",
    pre_attn_norm="transformer.layers.{}.attn_norm",
    attn_qkv_proj="transformer.layers.{}.attn.qkv_proj",
    attn_key_norm="transformer.layers.{}.attn.k_norm",
    attn_query_norm="transformer.layers.{}.attn.q_norm",
    attn_output_proj="transformer.layers.{}.attn.out_proj",
    pre_ff_norm="transformer.layers.{}.ffn_norm",
    ff_gate_up_proj="transformer.layers.{}.ffn.proj_1",
    ff_down_proj="transformer.layers.{}.ffn.proj_2",
    final_norm="transformer.norm",
)

ATTN_NUM_HEADS = [
    16,
    16,
    16,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    28,
    28,
    28,
    28,
    28,
    28,
    32,
    32,
    32,
    32,
]

ATTN_NUM_QUERY_GROUPS = [
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
]

FF_INTERMEDIATE_SIZES = [
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
    4608,
    4608,
    5120,
    5376,
    5632,
    5888,
    6144,
    6400,
    6656,
    6912,
    7168,
    7424,
    7680,
    7936,
    8192,
]


class OpenELM(nn.Module):

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()
    self.config = config
    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(self._get_model_config_for_layer(i))
        for i in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, config.lm_head_use_bias
    )
    # OpenELM re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data
    # Construct caches.
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(config.attn_config.rotary_percentage * config.attn_config.head_dim),
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

  def _get_model_config_for_layer(self, idx: int) -> cfg.ModelConfig:
    result = copy.deepcopy(self.config)
    result.attn_config.num_heads = ATTN_NUM_HEADS[idx]
    result.attn_config.num_query_groups = ATTN_NUM_QUERY_GROUPS[idx]
    result.ff_config.intermediate_size = FF_INTERMEDIATE_SIZES[idx]
    return result

  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    _, T = idx.size()
    assert (
        self.config.max_seq_len >= T
    ), f"Cannot forward sequence of length {T}, max seq length is only {self.config.max_seq_len}"
    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.kv_cache_max]
    # forward the model itself
    x = self.tok_embedding(idx)  # token embeddings of shape (b, t, n_embd)
    # TODO(mbrenon): gemma does `x * (self.config.embedding_dim**0.5)` here ?
    for _, block in enumerate(self.transformer_blocks):
      x = block(x, (cos, sin), mask, input_pos)
    x = self.final_norm(x)
    res = self.lm_head(x)  # (b, t, vocab_size)w
    return res


# Values set to -1 are dynamically updated in _get_model_config_for_layer(idx)
def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  attn_config = cfg.AttentionConfig(
      num_heads=-1,
      head_dim=64,
      num_query_groups=4,
      rotary_percentage=1.0,
      query_norm_config=norm_config,
      key_norm_config=norm_config,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationType.SILU,
      intermediate_size=-1,
  )
  # TODO(mbrenon): check if these fields should be set to something else
  # than the default:
  # - enable_hlfb
  # - relative_attention
  config = cfg.ModelConfig(
      vocab_size=32000,
      num_layers=28,
      max_seq_len=2048,
      embedding_dim=2048,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      pre_ff_norm_config=norm_config,
      final_norm_config=norm_config,
      kv_cache_max_len=kv_cache_max_len,
  )
  return config


def get_fake_model_config_for_test() -> cfg.ModelConfig:
  config = get_model_config()
  config.num_layers = 2
  return config


def build_model(checkpoint_path, **kwargs) -> nn.Module:
  config = get_model_config(**kwargs)
  model = OpenELM(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # lm_head re-uses existing weights, so we need to set strict mode to False.
  loader.load(model, strict=False)
  return model


def define_and_run() -> None:
  kv_cache_max_len = 1024
  checkpoint_path = os.path.join(
      Path.home(), "models/open_elm_1_1b_instruct/model.safetensors"
  )
  model = build_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.long, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len)
  print("running an inference")
  print(model.forward(tokens, input_pos))


if __name__ == "__main__":
  define_and_run()
