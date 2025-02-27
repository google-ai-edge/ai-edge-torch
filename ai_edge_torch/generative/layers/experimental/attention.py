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

"""Common building blocks for a GPU-specific Attention layer.

This is a temporary implemenation for the GPU. It is subject to change/removal
at any time.
"""

from typing import Optional, Tuple, Union

from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import lora as lora_utils
from ai_edge_torch.generative.layers.experimental import kv_cache as kv_utils
from ai_edge_torch.generative.layers.experimental import scaled_dot_product_attention as sdpa
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
import torch
from torch import nn


class TransformerBlock(nn.Module):

  def __init__(
      self,
      config: cfg.TransformerBlockConfig,
      model_config: cfg.ModelConfig,
  ) -> None:
    """Initialize an instance of the TransformerBlock.

    Args:
      config (cfg.TransformerBlockConfig): the configuration object for this
        transformer block.
      model_config (cfg.ModelConfig): the configuration object for the model
        this transformer block belongs to.
    """
    super().__init__()
    self.pre_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.pre_attention_norm_config,
    )
    self.atten_func = CausalSelfAttention(
        model_config.embedding_dim,
        config.attn_config,
        model_config.enable_hlfb,
    )
    self.post_atten_norm = builder.build_norm(
        model_config.embedding_dim,
        config.post_attention_norm_config,
    )
    self.ff = builder.build_ff(model_config.embedding_dim, config.ff_config)
    self.config = config

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: kv_utils.KVCacheEntryBase = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntryBase]]:
    """Forward function of the TransformerBlock.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntryBase): the optional kv cache entry.
      lora (LoRAEntry): the optional lora entry.

    Returns:
      output activation from this transformer block, and updated kv cache (if
      passed in).
    """
    kv = None
    if self.config.parallel_residual:
      x_norm = self.pre_atten_norm(x)
      atten_func_out = self.atten_func(
          x_norm, rope, mask, input_pos, kv_cache, lora
      )
      if kv_cache is None:
        attn_out = atten_func_out
      else:
        attn_out, kv = atten_func_out
      ff_out = self.ff(x_norm)
      output = x + attn_out + ff_out
    else:
      x_norm = self.pre_atten_norm(x)
      atten_func_out = self.atten_func(
          x_norm, rope, mask, input_pos, kv_cache, lora
      )
      if kv_cache is None:
        attn_out = atten_func_out
      else:
        attn_out, kv = atten_func_out
      x = x + attn_out
      x_norm = self.post_atten_norm(x)
      output = x + self.ff(x_norm)

    return output if kv is None else (output, kv)


class CausalSelfAttention(nn.Module):

  def __init__(
      self,
      dim: int,
      config: cfg.AttentionConfig,
      enable_hlfb: bool,
  ) -> None:
    """Initialize an instance of CausalSelfAttention.

    Args:
      dim (int): causal attention's input/output dimmension.
      config (cfg.AttentionConfig): attention specific configurations.
      enable_hlfb (bool): whether hlfb is enabled or not.
    """
    super().__init__()
    self.kv_cache = None
    qkv_shape = (
        config.num_heads + 2 * config.num_query_groups
    ) * config.head_dim
    output_shape = config.num_heads * config.head_dim
    # Key, query, value projections for all heads.
    self.qkv_projection = nn.Linear(dim, qkv_shape, bias=config.qkv_use_bias)
    self.output_projection = nn.Linear(
        output_shape, dim, bias=config.output_proj_use_bias
    )
    self.query_norm = builder.build_norm(
        config.head_dim, config.query_norm_config
    )
    self.key_norm = builder.build_norm(config.head_dim, config.key_norm_config)
    self.config = config
    self.enable_hlfb = enable_hlfb
    self.sdpa_func = sdpa.scaled_dot_product_attention

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntryBase] = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntryBase]]:
    """Forward function of the CausalSelfAttention layer, which can support

       MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntryBase): the KV cache entry corresponding to this
        module.
      lora (LoRAEntry): the optional lora entry.

    Returns:
      output activation from this self attention layer, and the updated
        KV Cach Entry (if passed in).
    """
    # Batch size, sequence length, embedding dimensionality.
    B, T, E = x.size()

    qkv = self.qkv_projection(x)

    # Assemble into a number of query groups to support MHA, MQA and GQA.
    q_per_kv = self.config.num_heads // self.config.num_query_groups
    # Each group has >=1 queries, 1 key, and 1 value.
    if self.config.qkv_transpose_before_split:
      qkv = qkv.view(B, T, -1, self.config.head_dim)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.num_query_groups,
              self.config.num_query_groups,
              self.config.num_query_groups,
          ),
          dim=-2,
      )
    else:
      qkv = qkv.view(B, T, self.config.num_query_groups, -1)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.head_dim,
              self.config.head_dim,
              self.config.head_dim,
          ),
          dim=-1,
      )

    if lora is not None:
      q += lora_utils.apply_lora(x, lora.attention.query, shape=q.shape)
      k += lora_utils.apply_lora(x, lora.attention.key, shape=k.shape)
      v += lora_utils.apply_lora(x, lora.attention.value, shape=v.shape)

    q = self.query_norm(q)
    k = self.key_norm(k)

    q = q.reshape(B, T, -1, self.config.head_dim)
    k = k.reshape(B, T, -1, self.config.head_dim)
    v = v.reshape(B, T, -1, self.config.head_dim)

    if rope is not None:
      # Compute rotary positional embedding for query and key.
      n_elem = int(self.config.rotary_percentage * self.config.head_dim)
      cos, sin = rope
      q, k = rotary_pos_emb.apply_rope_inline(q, k, cos, sin)

    # Transpose k/v to specific layout for GPU implementation.
    b, _, n, h = q.shape
    g = n // self.config.num_query_groups
    # btnh -> bnth -> b(kg)th -> 1(bk)(gt)h
    q = q.permute(0, 2, 1, 3).reshape(
        1, b * self.config.num_query_groups, g * T, h
    )

    k = k.permute(0, 2, 1, 3).reshape(
        1, -1, T, self.config.head_dim
    )  # 1, bk, s, h
    v = v.permute(0, 2, 3, 1).reshape(
        1, -1, self.config.head_dim, T
    )  # 1, bk, h, s

    if kv_cache is not None:
      kv_cache = kv_utils.update(kv_cache, input_pos, k, v)
      k, v = kv_cache.k_cache, kv_cache.v_cache

    sdpa_out = self.sdpa_func(
        kv_cache,
        q,
        k,
        v,
        self.config.head_dim,
        mask=mask,
        softcap=self.config.logit_softcap,
    )  # 1, bk, gt, h
    sdpa_out = (
        sdpa_out.reshape(B, -1, T, h).permute(0, 2, 1, 3).reshape(B, T, -1)
    )

    # Compute the output projection.
    y = self.output_projection(sdpa_out)
    if lora is not None:
      y += lora_utils.apply_lora(sdpa_out, lora.attention.output)

    return y if kv_cache is None else (y, kv_cache)
