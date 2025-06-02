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

"""Example of building a Phi-4 model up to 4K tokens, not to 128K tokens."""

from functools import partial
import math
from typing import Callable, Dict, Tuple

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.gate_up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    attn_fused_qkv_proj="model.layers.{}.self_attn.qkv_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
)

# max_position_embeddings / original_max_position_embeddings in Phi-4 config.
ROPE_SCALE_FACTOR = 32

# ROPE short factor in Phi-4 config. According to LOPE paper and its code in
# https://github.com/microsoft/LongRoPE, these values had been searched with
# min=1.0, step-0.01 to optimize the errors of sample dataset.
ROPE_SHORT_FACTOR = [1.0] * 48


def _build_phi4_rope(
    input_pos: int,
    n_elem: int,
    base: int,
    condense_ratio: int,
    dtype: torch.dtype,
    device: torch.device,
    theta_factors: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes Rotary Positional Embeddings for Phi-4 model.

  It's a modified version of attn_utils.build_rope_cache with additional
  arguments for Phi-4 model. It precompute Rotary Positional Embedding Sin and
  Cos values with scaling factors for quick lookup during the inference.

  Args:
      input_pos (torch.Tensor): the given input sequence positions
      n_elem (int): Each sequence's dimmension.
      base (int, optional): Rope base value.
      condense_ratio (int, optional): The ratio by which sequence indicies are
        condensed.
      dtype (torch.dtype, optional): Output tensor's data type.
      device (torch.device, optional): Output tensor's data type.
      theta_factors (torch.Tensor, optional): A tensor of shape (n_elem,) used
        to scale the theta values.
      scale (float, optional): A float used to scale the rope values.

  Returns:
      Tuple[torch.Tensor, torch.Tensor]: Rope's Cosine and Sine waves.
  """
  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2).float() / n_elem))
  theta = theta / theta_factors
  seq_idx = input_pos / condense_ratio
  idx_theta = torch.outer(seq_idx, theta)
  cos = torch.cos(idx_theta).to(dtype=dtype, device=device) * scale
  sin = torch.sin(idx_theta).to(dtype=dtype, device=device) * scale
  return cos, sin


class Phi4Mini(model_builder.DecoderOnlyModel):
  """A Phi-4 model built from the Edge Generative API layers."""
  pass


def get_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Phi-4 model."""
  attn_config = cfg.AttentionConfig(
      num_heads=24,
      head_dim=128,
      num_query_groups=8,
      rotary_base=10000,
      rotary_percentage=0.75,
      qkv_transpose_before_split=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU_GLU),
      intermediate_size=8192,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )

  max_seq_len = 4096
  # Create the RoPE callable
  build_rope = partial(
      _build_phi4_rope,
      condense_ratio=1,
      dtype=torch.float32,
      device=torch.device("cpu"),
      theta_factors=torch.tensor(ROPE_SHORT_FACTOR),
      scale=math.sqrt(1 + math.log(ROPE_SCALE_FACTOR) / math.log(max_seq_len)),
  )

  config = cfg.ModelConfig(
      vocab_size=200064,
      num_layers=32,
      max_seq_len=max_seq_len,
      embedding_dim=3072,
      block_configs=block_config,
      final_norm_config=norm_config,
      build_rope=build_rope,
  )
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_model_config()
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 256
  # Phi-4 has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  return config


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> torch.nn.Module:
  """Instantiates the model instance and load checkpoint if provided."""
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_model_config(),
      tensor_names=TENSOR_NAMES,
      model_class=Phi4Mini,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )
