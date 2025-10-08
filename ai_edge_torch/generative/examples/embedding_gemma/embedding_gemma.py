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
"""EmbeddingGemma-300M model implementation."""

import math
import os
from typing import Callable, Dict

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import attention_utils
from ai_edge_torch.generative.layers import normalization as norm
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import loader
from safetensors.torch import load_file
import torch
from torch import nn


class EmbeddingGemma(nn.Module):
  """EmbeddingGemma-300M model implementation."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()
    self.config = config
    self.embedder = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.transformer_blocks = nn.ModuleList([
        attention.TransformerBlock(block_config, config)
        for block_config in config.block_configs
    ])
    self.dense1 = nn.Linear(config.embedding_dim, 3072, bias=False)
    self.dense2 = nn.Linear(3072, config.embedding_dim, bias=False)

  def _prepare_attention_mask(self, attention_mask, input_shape, dtype, device):
    """Creates a padding attention mask."""
    batch_size, seq_len = input_shape
    if attention_mask is None:
      return torch.zeros((batch_size, 1, 1, seq_len), dtype=dtype, device=device)
    padding_mask = torch.where(
        attention_mask == 0, torch.finfo(dtype).min, 0.0
    )
    return padding_mask[:, None, None, :]

  def mean_pool(self, last_hidden_states, attention_mask):
    """Mean pooling of hidden states, ignoring padding tokens."""
    masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
    sum_hidden_states = masked_hidden_states.sum(dim=1)
    count = attention_mask.sum(dim=1).unsqueeze(-1)
    return sum_hidden_states / (count + 1e-9)

  def forward(
      self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    batch_size, seq_len = tokens.shape
    if attention_mask is None:
      attention_mask = torch.ones(batch_size, seq_len, device=tokens.device)

    x = self.embedder(tokens)
    x = x * math.sqrt(self.config.embedding_dim)

    positions = torch.arange(0, seq_len, device=tokens.device)
    attn_mask = self._prepare_attention_mask(
        attention_mask, (batch_size, seq_len), x.dtype, x.device
    )
    rope_cos, rope_sin = attention_utils.build_rope_cache(
        size=self.config.max_seq_len,
        dim=self.config.block_configs[0].attn_config.head_dim,
        base=self.config.block_configs[0].attn_config.rotary_base,
        dtype=x.dtype,
        device=x.device,
    )
    rope = (rope_cos[positions], rope_sin[positions])

    for block in self.transformer_blocks:
      x = block(x, rope, attn_mask, kv_cache=None)

    pooled_x = self.mean_pool(x, attention_mask)
    pooled_x = self.dense1(pooled_x)
    pooled_x = self.dense2(pooled_x)
    normalized_x = torch.nn.functional.normalize(pooled_x, p=2, dim=1)
    return normalized_x


def get_model_config() -> cfg.ModelConfig:
  """Returns the model config for EmbeddingGemma-300M."""
  attn_config = cfg.AttentionConfig(
      num_heads=3,
      head_dim=256,
      num_query_groups=1,  # MQA
      rotary_base=1000000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=1152,
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
      parallel_residual=False,
  )
  config = cfg.ModelConfig(
      vocab_size=262144,
      num_layers=24,
      max_seq_len=8192,
      embedding_dim=768,
      block_configs=[block_config] * 24,
      final_norm_config=norm_config,
  )
  return config


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
) -> nn.Module:
  """Builds the EmbeddingGemma-300M model."""
  config = get_model_config()
  model = EmbeddingGemma(config)
  state_dict = {}

  has_sub_dirs = os.path.exists(
      os.path.join(checkpoint_path, "2_Dense", "model.safetensors")
  ) or os.path.exists(
      os.path.join(checkpoint_path, "2_Dense", "pytorch_model.bin")
  )

  if has_sub_dirs:
    try:
      weights = loader.load_safetensors(checkpoint_path)
      weights_dense1 = load_file(
          os.path.join(checkpoint_path, "2_Dense", "model.safetensors")
      )
      weights_dense2 = load_file(
          os.path.join(checkpoint_path, "3_Dense", "model.safetensors")
      )
      state_dict["dense1.weight"] = weights_dense1["linear.weight"]
      state_dict["dense2.weight"] = weights_dense2["linear.weight"]
    except Exception:
      weights = loader.load_pytorch_statedict(checkpoint_path)
      weights_dense1 = torch.load(
          os.path.join(checkpoint_path, "2_Dense", "pytorch_model.bin")
      )
      weights_dense2 = torch.load(
          os.path.join(checkpoint_path, "3_Dense", "pytorch_model.bin")
      )
      state_dict["dense1.weight"] = weights_dense1["linear.weight"]
      state_dict["dense2.weight"] = weights_dense2["linear.weight"]
  else:
    try:
      weights = loader.load_safetensors(checkpoint_path)
      state_dict["dense1.weight"] = weights["dense1.weight"]
      state_dict["dense2.weight"] = weights["dense2.weight"]
    except Exception:
      weights = loader.load_pytorch_statedict(checkpoint_path)
      state_dict["dense1.weight"] = weights["dense1.weight"]
      state_dict["dense2.weight"] = weights["dense2.weight"]

  state_dict["embedder.weight"] = weights["embed_tokens.weight"]

  for i in range(config.num_layers):
    layer_prefix = f"layers.{i}"
    tb_prefix = f"transformer_blocks.{i}"
    # Norms
    state_dict[f"{tb_prefix}.pre_atten_norm.weight"] = weights[
        f"{layer_prefix}.input_layernorm.weight"
    ]
    state_dict[f"{tb_prefix}.post_atten_norm.weight"] = weights[
        f"{layer_prefix}.post_attention_layernorm.weight"
    ]
    # Attention
    q = weights[f"{layer_prefix}.self_attn.q_proj.weight"]
    k = weights[f"{layer_prefix}.self_attn.k_proj.weight"]
    v = weights[f"{layer_prefix}.self_attn.v_proj.weight"]
    state_dict[f"{tb_prefix}.atten_func.qkv_projection.weight"] = torch.cat(
        [q, k, v], dim=0
    )
    state_dict[f"{tb_prefix}.atten_func.output_projection.weight"] = weights[
        f"{layer_prefix}.self_attn.o_proj.weight"
    ]
    # Feed-forward
    state_dict[f"{tb_prefix}.ff.w1.weight"] = weights[
        f"{layer_prefix}.mlp.gate_proj.weight"
    ]
    state_dict[f"{tb_prefix}.ff.w3.weight"] = weights[
        f"{layer_prefix}.mlp.up_proj.weight"
    ]
    state_dict[f"{tb_prefix}.ff.w2.weight"] = weights[
        f"{layer_prefix}.mlp.down_proj.weight"
    ]

  model.load_state_dict(state_dict)
  return model
