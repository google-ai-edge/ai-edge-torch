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

"""Example of building a Phi-2 model."""

from typing import Callable, Dict
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.fc1",
    ff_down_proj="model.layers.{}.mlp.fc2",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.dense",
    pre_attn_norm="model.layers.{}.input_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.final_layernorm",
    lm_head="lm_head",
)


class Phi2(model_builder.DecoderOnlyModel):
  """A Phi-2 model built from the Edge Generative API layers."""
  pass


def get_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Phi-2 model."""
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=80,
      num_query_groups=32,
      rotary_base=10000,
      rotary_percentage=0.4,
      qkv_use_bias=True,
      output_proj_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=10240,
      use_bias=True,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.LAYER_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      parallel_residual=True,
  )
  config = cfg.ModelConfig(
      vocab_size=51200,
      num_layers=32,
      max_seq_len=2048,
      embedding_dim=2560,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_use_bias=True,
      lm_head_share_weight_with_embedding=False,
  )
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_model_config()
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 256
  # Phi-2 has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  return config


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_model_config(),
      tensor_names=TENSOR_NAMES,
      model_class=Phi2,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )
