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

"""Example of building a Gemma2 model."""

from typing import Callable, Dict, List, Optional, Tuple

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.utilities import export_config as export_cfg
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
    pre_ff_norm="model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="model.layers.{}.post_feedforward_layernorm",
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
    pre_ff_norm="model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="model.layers.{}.post_feedforward_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
)

TENSOR_NAMES_DICT = {
    "safetensors": TENSOR_NAMES_SEP_QKV,
    "kaggle": TENSOR_NAMES_FUSED_QKV,
}


class Gemma2Block(attention.TransformerBlock):

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: kv_utils.KVCacheEntry = None,
  ) -> Tuple[torch.Tensor, Optional[kv_utils.KVCacheEntry]]:
    """Forward function of the Gemma2Block.

    Exactly the same as TransformerBlock but we call the post-attention norm
    immediately after attention and not after the residual pointwise addition.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the optional kv cache entry.

    Returns:
      output activation from this transformer block, and updated kv cache (if
      passed in).
    """

    x_norm = self.pre_atten_norm(x)
    attn_out, kv = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)
    attn_out_norm = self.post_atten_norm(attn_out)
    x = x + attn_out_norm
    output = x + self.ff(x)
    return output, kv


class Gemma2(nn.Module):
  """A Gemma2 model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim,
        config.vocab_size,
        bias=config.lm_head_use_bias,
    )
    # Gemma2 re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        Gemma2Block(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.config = config
    self.build_mask_cache(mask_cache_size)

  def build_mask_cache(self, mask_cache_size: int):
    assert (
        mask_cache_size <= self.config.max_seq_len
    ), "Mask cache size must be less than or equal to the max seq length."
    if mask_cache_size <= 0:
      self.mask_cache = None
      self.sliding_window_mask_cache = None
      return
    self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size)
    # Gemma2 has same hyper parameters for each layer except for attention
    # types. Use the first layer.
    self.sliding_window_mask_cache = attn_utils.build_sliding_window_mask_cache(
        size=mask_cache_size,
        window_size=self.config.block_config(0).attn_config.sliding_window_size,
    )

  def get_attention_mask(
      self, attn_type: cfg.AttentionType, input_pos: torch.Tensor
  ) -> torch.Tensor:
    if attn_type == cfg.AttentionType.LOCAL_SLIDING:
      return self.sliding_window_mask_cache.index_select(2, input_pos)
    return self.mask_cache.index_select(2, input_pos)

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    # token embeddings of shape (b, t, n_embd)
    input_embeds = self.tok_embedding(tokens)
    # RoPE parameters are the same for all blocks. Use the first layer.
    attn_config = self.config.block_config(0).attn_config
    n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
    rope = rotary_pos_emb.build_rope(input_pos, n_elem, attn_config.rotary_base)
    if mask is None:
      assert self.mask_cache is not None, "Mask cache must be built."
      mask = [
          self.get_attention_mask(
              self.config.block_config(i).attn_config.attn_type, input_pos
          )
          for i in range(self.config.num_layers)
      ]

    return self._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, export_config
    )

  def _forward_with_embeds(
      self,
      input_embeds: torch.Tensor,
      rope: Tuple[torch.Tensor, torch.Tensor],
      mask: torch.Tensor | List[torch.Tensor],
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    """Forwards the model with input embeddings."""
    assert len(self.transformer_blocks) == len(kv_cache.caches), (
        "The number of transformer blocks and the number of KV cache entries"
        " must be the same."
    )

    if self.config.embedding_scale is not None:
      input_embeds = input_embeds * self.config.embedding_scale
    x = input_embeds
    updated_kv_entries = []
    for i, block in enumerate(self.transformer_blocks):
      mask_entry = mask[i] if isinstance(mask, list) else mask
      kv_entry = kv_cache.caches[i] if kv_cache else None
      x, kv_entry = block(x, rope, mask_entry, input_pos, kv_entry)
      if kv_entry:
        updated_kv_entries.append(kv_entry)
    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))

    if export_config is not None:
      if (
          torch.numel(input_pos) > 1
          and not export_config.output_logits_on_prefill
      ):
        return {"kv_cache": updated_kv_cache}

    x = self.final_norm(x)
    res = self.lm_head(x)  # (b, t, vocab_size)
    if self.config.final_logit_softcap is not None:
      res = res / self.config.final_logit_softcap
      res = torch.tanh(res)
      res = res * self.config.final_logit_softcap

    return {"logits": res, "kv_cache": updated_kv_cache}


def get_model_config_2b() -> cfg.ModelConfig:
  """Returns the model config for a Gemma2 2B model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=True
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=9216,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=8,
        head_dim=256,
        num_query_groups=4,
        rotary_base=10000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        # The safetensors from HF is not using the interleaved qkv format, so
        # we need to disable interleaving here in the model config.
        qkv_fused_interleaved=False,
        logit_softcap=50.0,
        sliding_window_size=4096,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if (idx + 1) % 2 == 0
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  num_layers = 26
  embedding_dim = 2304
  config = cfg.ModelConfig(
      vocab_size=256000,
      num_layers=num_layers,
      max_seq_len=8192,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=30.0,
  )
  return config


def get_fake_model_config() -> cfg.ModelConfig:
  config = get_model_config_2b()
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 256
  config.embedding_dim = 128
  config.embedding_scale = config.embedding_dim**0.5
  config.block_configs = config.block_configs[: config.num_layers]
  for block_config in config.block_configs:
    block_config.attn_config.num_heads = 4
    block_config.attn_config.head_dim = 64
    block_config.attn_config.sliding_window_size = 64
    block_config.ff_config.intermediate_size = 128
  return config


def build_2b_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  for tensor_names in TENSOR_NAMES_DICT.values():
    try:
      return model_builder.build_decoder_only_model(
          checkpoint_path=checkpoint_path,
          config=get_model_config_2b(),
          tensor_names=tensor_names,
          model_class=Gemma2,
          custom_loader=custom_loader,
          mask_cache_size=mask_cache_size,
      )
    except KeyError as _:
      continue
