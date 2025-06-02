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

"""Example of building Qwen 2.5 models."""

from typing import Callable, Dict
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import torch
from torch import nn

TENSOR_NAMES = model_builder.TENSOR_NAMES


class Qwen(model_builder.DecoderOnlyModel):
  """A Qwen model built from the Edge Generative API layers."""
  pass


def get_3b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 3B model."""
  attn_config = cfg.AttentionConfig(
      num_heads=16,
      head_dim=128,
      num_query_groups=2,
      rotary_base=1000000,
      rotary_percentage=1.0,
      qkv_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=11008,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-06
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
      max_seq_len=32768,
      embedding_dim=2048,
      block_configs=block_config,
      final_norm_config=norm_config,
  )
  return config


def get_1_5b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 1B model."""
  config = get_3b_model_config()
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 12
  block_config.ff_config.intermediate_size = 8960
  config.num_layers = 28
  config.embedding_dim = 1536
  return config


def get_0_5b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 0.5B model."""
  config = get_3b_model_config()
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 14
  block_config.attn_config.head_dim = 64
  block_config.ff_config.intermediate_size = 4864
  config.num_layers = 24
  config.embedding_dim = 896
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_3b_model_config()
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
      model_class=Qwen,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )


def build_3b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return _build_model(
      checkpoint_path, get_3b_model_config(), custom_loader, mask_cache_size
  )


def build_1_5b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return _build_model(
      checkpoint_path, get_1_5b_model_config(), custom_loader, mask_cache_size
  )


def build_0_5b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return _build_model(
      checkpoint_path, get_0_5b_model_config(), custom_loader, mask_cache_size
  )
