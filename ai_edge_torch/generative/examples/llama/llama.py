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

"""Example of building Llama 3.2 models."""

from functools import partial
import math
from typing import Tuple

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import torch

TENSOR_NAMES = model_builder.TENSOR_NAMES


def _build_llama3_rope_cache(
    input_pos: torch.Tensor,
    n_elem: int,
    base: int,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes Rotary Positional Embeddings for Llama 3.2 model.

  It's a modified version of attn_utils.build_rope_cache with additional
  arguments for Llama 3.2 model. It precomputes Rotary Positional Embedding Sin
  and Cos values with scaling factors for quick lookup during the inference.

  Reference:
  https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L307

  Args:
      input_pos (torch.Tensor): the given input sequence positions
      n_elem (int): Each sequence's dimmension.
      base (int): Rope base value.
      factor (float): Factor to scale theta down for tokens in long range in the
        sequence.
      low_freq_factor (float): Factor to determine if tokens are in long range
        in the sequence.
      high_freq_factor (float): Factor to determine if tokens are in short range
        in the sequence.
      max_seq_len (int): The original token sequence length before extending
        ROPE to support longer sequence.

  Returns:
      Tuple[torch.Tensor, torch.Tensor]: Rope's Cosine and Sine waves.
  """
  theta = 1.0 / (base ** (torch.arange(0, n_elem, 2).float() / n_elem))
  low_freq_wavelen = max_seq_len / low_freq_factor
  high_freq_wavelen = max_seq_len / high_freq_factor
  wavelen = 2 * math.pi / theta
  # wavelen < high_freq_wavelen: do nothing
  # wavelen > low_freq_wavelen: divide by factor
  theta = torch.where(wavelen > low_freq_wavelen, theta / factor, theta)
  # otherwise: interpolate between the two, using a smooth factor
  smooth_factor = (max_seq_len / wavelen - low_freq_factor) / (
      high_freq_factor - low_freq_factor
  )
  smoothed_theta = (1 - smooth_factor) * theta / factor + smooth_factor * theta
  is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
  theta = torch.where(is_medium, smoothed_theta, theta)

  idx_theta = torch.outer(input_pos, theta)
  cos = torch.cos(idx_theta).to(dtype=torch.float32, device=torch.device("cpu"))
  sin = torch.sin(idx_theta).to(dtype=torch.float32, device=torch.device("cpu"))
  return cos, sin


class Llama(model_builder.DecoderOnlyModel):
  """A Llama model built from the Edge Generative API layers.

  Llama 3.2 shares the same architecture as TinyLlama except ROPE calculation.
  """

  def forward(self, tokens, input_pos, kv_cache, **kwargs):
    if kwargs.get("rope") is None:
      # ROPE parameters for all attn_configs are the same. Take the first one.
      attn_config = self.config.block_config(0).attn_config
      kwargs["rope"] = _build_llama3_rope_cache(
          input_pos=input_pos,
          n_elem=int(attn_config.rotary_percentage * attn_config.head_dim),
          base=attn_config.rotary_base,
          factor=32.0,
          low_freq_factor=1.0,
          high_freq_factor=4.0,
          max_seq_len=self.config.max_seq_len,
      )
    return super().forward(tokens, input_pos, kv_cache, **kwargs)


def get_1b_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Llama 3.2-1B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a SmolLM model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=64,
      num_query_groups=8,
      rotary_base=500000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=8192,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=128256,
      num_layers=16,
      max_seq_len=8192,
      embedding_dim=2048,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_3b_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Llama 3.2-3B model."""
  config = get_1b_model_config(kv_cache_max_len)
  # Llama 3.2 has only one block config.
  attn_config = config.block_config(0).attn_config
  attn_config.num_heads = 24
  attn_config.head_dim = 128
  config.num_layers = 28
  config.embedding_dim = 3072
  return config


def get_fake_model_config(**kwargs) -> cfg.ModelConfig:
  config = get_1b_model_config(**kwargs)
  config.vocab_size = 128
  config.num_layers = 2
  # SmolLM has only one block config.
  config.block_config(0).ff_config.intermediate_size = 64
  return config


def _build_model(
    checkpoint_path: str, config: cfg.ModelConfig
) -> torch.nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=config,
      tensor_names=TENSOR_NAMES,
      model_class=Llama,
  )


def build_1b_model(checkpoint_path: str, **kwargs) -> torch.nn.Module:
  return _build_model(checkpoint_path, get_1b_model_config(**kwargs))


def build_3b_model(checkpoint_path: str, **kwargs) -> torch.nn.Module:
  return _build_model(checkpoint_path, get_3b_model_config(**kwargs))
