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

"""Example of building Qwen 3.0 models."""

from typing import Callable, Dict
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import loader as loading_utils
from ai_edge_torch.generative.utilities import model_builder
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_query_norm="model.layers.{}.self_attn.q_norm",
    attn_key_norm="model.layers.{}.self_attn.k_norm",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head="lm_head",
)


class Qwen3(model_builder.DecoderOnlyModel):
  """A Qwen3 model built from the Edge Generative API layers."""
  pass


def get_4b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 3.0 4B model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-06
  )
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=128,
      num_query_groups=8,
      query_norm_config=norm_config,
      key_norm_config=norm_config,
      rotary_base=1000000,
      rotary_percentage=1.0,
      qkv_use_bias=False,
      qkv_transpose_before_split=True,
      qkv_fused_interleaved=False,  # No interleaved qkv projection.
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=9728,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=151936,
      num_layers=36,
      max_seq_len=40960,
      embedding_dim=2560,
      block_configs=block_config,
      final_norm_config=norm_config,
  )
  return config


def get_1_7b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 3.0 1.7B model."""
  config = get_4b_model_config()
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 16
  block_config.attn_config.head_dim = 128
  block_config.ff_config.intermediate_size = 6144
  config.num_layers = 28
  config.embedding_dim = 2048
  return config


def get_0_6b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 3.0 0.6B model."""
  config = get_4b_model_config()
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 16
  block_config.attn_config.head_dim = 128
  block_config.ff_config.intermediate_size = 3072
  config.num_layers = 28
  config.embedding_dim = 1024
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_4b_model_config()
  config.vocab_size = 128
  config.num_layers = 2
  # Qwen has only one block config.
  config.block_config(0).ff_config.intermediate_size = 64
  return config


def _build_model(
    checkpoint_path: str,
    config: cfg.ModelConfig,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=config,
      tensor_names=TENSOR_NAMES,
      model_class=Qwen3,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )


def build_4b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return _build_model(
      checkpoint_path, get_4b_model_config(), custom_loader, mask_cache_size
  )


def build_1_7b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return _build_model(
      checkpoint_path, get_1_7b_model_config(), custom_loader, mask_cache_size
  )


def build_0_6b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return _build_model(
      checkpoint_path, get_0_6b_model_config(), custom_loader, mask_cache_size
  )
