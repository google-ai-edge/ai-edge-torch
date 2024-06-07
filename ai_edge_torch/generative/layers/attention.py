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
# Common building blocks for Attention layer.

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

import ai_edge_torch.generative.layers.builder as builder
from ai_edge_torch.generative.layers.kv_cache import KVCache
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.layers.scaled_dot_product_attention import scaled_dot_product_attention  # NOQA
from ai_edge_torch.generative.layers.scaled_dot_product_attention import scaled_dot_product_attention_with_hlfb  # NOQA


class TransformerBlock(nn.Module):

  def __init__(self, config: cfg.ModelConfig) -> None:
    """Initialize an instance of the TransformerBlock.

    Args:
      config (cfg.ModelConfig): the configuration object
        for this transformer block.
    """

    super().__init__()
    self.pre_atten_norm = builder.build_norm(
        config.embedding_dim, config.pre_attention_norm_config
    )
    self.atten_func = CausalSelfAttention(
        config.embedding_dim,
        config.attn_config,
        config.kv_cache_max,
        config.enable_hlfb,
    )
    self.pre_ff_norm = builder.build_norm(
        config.embedding_dim, config.pre_ff_norm_config
    )
    self.ff = builder.build_ff(config.embedding_dim, config.ff_config)
    self.config = config

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the TransformerBlock.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.

    Returns:
      output activation from this transformer block.
    """

    if self.config.parallel_residual:
      x_norm = self.pre_atten_norm(x)
      attn_out = self.atten_func(x_norm, rope, mask, input_pos)
      ff_out = self.ff(x_norm)
      output = x + attn_out + ff_out
    else:
      x_norm = self.pre_atten_norm(x)
      attn_out = self.atten_func(x_norm, rope, mask, input_pos)
      x = x + attn_out
      x_norm = self.pre_ff_norm(x)
      output = x + self.ff(x_norm)

    return output


class CausalSelfAttention(nn.Module):

  def __init__(
      self,
      dim: int,
      config: cfg.AttentionConfig,
      kv_cache_max: int,
      enable_hlfb: bool,
  ) -> None:
    """Initialize an instance of CausalSelfAttention.

    Args:
      dim (int): causal attention's input/output dimmension.
      config (cfg.AttentionConfig): attention specific configurations.
      kv_cache_max (int): determines the size of the KV Cache buffer, if enabled.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__()
    self.head_dim = dim // config.num_heads
    shape = (config.num_heads + 2 * config.num_query_groups) * self.head_dim
    # Key, query, value projections for all heads.
    self.qkv_projection = nn.Linear(dim, shape, bias=config.qkv_use_bias)
    self.output_projection = nn.Linear(dim, dim, bias=config.output_proj_use_bias)
    self.config = config
    self.kv_cache = None

    # Build a k/v cache with size (batch_size, kv_cache_max, n_heads, head_dim).
    # Now only supports batch_size of 1.
    # TODO(haoliang): support batch_size greater than 1.
    if config.enable_kv_cache:
      self.kv_cache = KVCache(
          1,
          kv_cache_max,
          config.num_query_groups,
          self.head_dim,
          enable_hlfb,
      )

    if enable_hlfb:
      self.sdpa_func = scaled_dot_product_attention_with_hlfb
    else:
      self.sdpa_func = scaled_dot_product_attention

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the CausalSelfAttention layer, which can support
       MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.

    Returns:
      output activation from this self attention layer.
    """
    # Batch size, sequence length, embedding dimensionality.
    B, T, E = x.size()
    assert B == 1, "Currently only batch_size = 1 is supported."

    qkv = self.qkv_projection(x)

    # Assemble into a number of query groups to support MHA, MQA and GQA.
    q_per_kv = self.config.num_heads // self.config.num_query_groups
    total_qkv = q_per_kv + 2  # Each group has >=1 queries, 1 key, and 1 value.
    if self.config.qkv_transpose_before_split:
      qkv = qkv.view(
          B, T, total_qkv, self.config.num_query_groups, self.head_dim
      )  # (B, T, total_qkv, num_query_groups, head_dim)
      qkv_axis = -3
    else:
      qkv = qkv.view(
          B, T, self.config.num_query_groups, total_qkv, self.head_dim
      )  # (B, T, num_query_groups, total_qkv, head_dim)
      qkv_axis = -2

    # Split batched computation into three.
    q, k, v = qkv.split((q_per_kv, 1, 1), dim=qkv_axis)
    q = q.reshape(B, T, -1, self.head_dim)
    k = k.reshape(B, T, -1, self.head_dim)
    v = v.reshape(B, T, -1, self.head_dim)

    # Compute rotary positional embedding for query and key.
    n_elem = int(self.config.rotary_percentage * self.head_dim)
    if n_elem > 0:
      cos, sin = rope
      q_roped = rotary_pos_emb.apply_rope(
          q[..., :n_elem], cos.repeat(1, 2), sin.repeat(1, 2)
      )
      k_roped = rotary_pos_emb.apply_rope(
          k[..., :n_elem], cos.repeat(1, 2), sin.repeat(1, 2)
      )
      q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
      k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

    if self.kv_cache is not None:
      # TODO(haoliang): Handle when execeeding max sequence length.
      k, v = self.kv_cache.update_cache(input_pos, k, v)

    y = self.sdpa_func(q, k, v, self.head_dim, mask=mask)
    y = y.reshape(B, T, E)

    # Compute the output projection.
    y = self.output_projection(y)
    return y


class SelfAttention(CausalSelfAttention):
  """Non-causal Self Attention module, which is equivalent to CausalSelfAttention without mask."""

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      input_pos: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the SelfAttention layer, which can support MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      input_pos (torch.Tensor): the optional input position tensor.

    Returns:
      output activation from this self attention layer.
    """
    B, T, _ = x.size()
    return super().forward(
        x, rope=rope, mask=torch.zeros((B, T), dtype=torch.float32), input_pos=input_pos
    )
