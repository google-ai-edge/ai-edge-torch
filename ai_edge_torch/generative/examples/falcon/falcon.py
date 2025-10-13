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

"""Falcon-1B model implementation."""

from typing import Callable, Dict
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import model_builder
import torch
from torch import nn

TENSOR_NAMES = loader.ModelLoader.TensorNames(
    embedding="transformer.word_embeddings",
    final_norm="transformer.ln_f",
    pre_attn_norm="transformer.h.{}.input_layernorm",
    post_attn_norm="transformer.h.{}.post_attention_layernorm",
    attn_fused_qkv_proj="transformer.h.{}.self_attention.query_key_value",
    attn_output_proj="transformer.h.{}.self_attention.dense",
    ff_up_proj="transformer.h.{}.mlp.dense_h_to_4h",
    ff_down_proj="transformer.h.{}.mlp.dense_4h_to_h",
    lm_head="lm_head",
)


class Falcon(model_builder.DecoderOnlyModel):
  """A Falcon-1B model built from the Edge Generative API layers."""
  pass


def get_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Falcon-1B model."""
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=64,
      num_query_groups=32,  # Multi-Head Attention
      use_alibi=True,
      rotary_percentage=0.0,
      qkv_use_bias=True,
      output_proj_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,  # Falcon uses a standard MLP
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU),
      intermediate_size=8192,  # 4 * embedding_dim
      use_bias=True,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.LAYER_NORM, epsilon=1e-5, use_bias=True
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      parallel_residual=False,  # parallel_attn=False in config
  )
  config = cfg.ModelConfig(
      vocab_size=50304,
      num_layers=24,
      max_seq_len=2048,
      embedding_dim=2048,
      block_configs=[block_config] * 24,  # All layers are the same
      final_norm_config=norm_config,
  )
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_model_config()
  config.vocab_size = 128
  config.num_layers = 2
  config.block_configs[0].ff_config.intermediate_size = 64
  return config


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  """Builds the Falcon-1B model."""
  # Using default TENSOR_NAMES from model_builder for now.
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_model_config(),
      # Uncomment for testing with a smaller model
      # config=get_fake_model_config(),
      tensor_names=TENSOR_NAMES,
      model_class=Falcon,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )
