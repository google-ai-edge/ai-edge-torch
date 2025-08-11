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

"""Example of building a Gemma1 model."""

from typing import Callable, Dict

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

TENSOR_NAMES_FUSED_QKV = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_fused_qkv_proj="model.layers.{}.self_attn.qkv_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="embedder",
    final_norm="model.norm",
    lm_head=None,
)

TENSOR_NAMES_SEP_QKV = loading_utils.ModelLoader.TensorNames(
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
)

TENSOR_NAMES_DICT = {
    "safetensors": TENSOR_NAMES_SEP_QKV,
    "kaggle": TENSOR_NAMES_FUSED_QKV,
}

class Gemma1(model_builder.DecoderOnlyModel):
  """A Gemma1 model built from the Edge Generative API layers."""
  pass


def get_model_config_2b() -> cfg.ModelConfig:
  """Returns the model config for a Gemma 2B model."""
  attn_config = cfg.AttentionConfig(
      num_heads=8,
      head_dim=256,
      num_query_groups=1,
      rotary_base=10000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=16384,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=True
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  embedding_dim = 2048
  config = cfg.ModelConfig(
      vocab_size=256000,
      num_layers=18,
      max_seq_len=8192,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_use_bias=False,
  )
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_model_config_2b()
  # Gemma has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 256
  return config


def build_2b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:

  # A list to store the reasons for each failure
  key_errors = []

  for tensor_names in TENSOR_NAMES_DICT.values():
    try:
      return model_builder.build_decoder_only_model(
          checkpoint_path=checkpoint_path,
          config=get_model_config_2b(),
          tensor_names=tensor_names,
          model_class=Gemma1,
          custom_loader=custom_loader,
          mask_cache_size=mask_cache_size,
      )
    except KeyError as ke:
      # Store the specific key that was missing for later
      key_errors.append(f"Missing key: {ke}")
      continue

  # If the loop finishes, raise an error with all the collected details
  error_details = "\n".join(key_errors)
  raise RuntimeError(
      "Failed to build model after trying all configurations. "
      f"Encountered the following errors:\n{error_details}"
  )
