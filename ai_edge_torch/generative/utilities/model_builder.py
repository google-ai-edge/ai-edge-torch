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

"""Utilities to be used for re-authoring transformer models."""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import lora as lora_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
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

TENSOR_NAMES_WITH_SEPARATE_LM_HEAD = copy.copy(TENSOR_NAMES)
TENSOR_NAMES_WITH_SEPARATE_LM_HEAD.lm_head = "lm_head"


@dataclass
class ExportConfig:
  """Model generating configuration settings."""

  # On prefill signatures, should the model produce logit output?
  # When False, only decode signatures will produce output.
  output_logits_on_prefill: bool = False
  # Attention masks given as inputs to the model.
  prefill_mask: Optional[torch.Tensor | List[torch.Tensor]] = None
  decode_mask: Optional[torch.Tensor | List[torch.Tensor]] = None
  # The KV Cache class for K and V buffers in attention.
  kvcache_cls: type = kv_utils.KVCache
  # The batch size of the decode signature.
  decode_batch_size: int = 1


class DecoderOnlyModel(nn.Module):
  """A simple decoder-only transformer model built from the Edge Generative API.

  This model is used for re-authoring. model_config is used to specify the
  details of model architecture and parameters.

  It assumes that the attention configs for ROPE, i.e. head_dim, rotary_base,
  and rotary_percentage are the same for all layers.
  """

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    if config.lm_head_share_weight_with_embedding:
      self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max,
    )
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      lora: Optional[lora_utils.LoRA] = None,
      export_config: Optional[ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    # token embeddings of shape (b, t, n_embd)
    input_embeds = self.tok_embedding(tokens)

    # ROPE parameters for all attn_configs are the same. Take the first one.
    attn_config = self.config.block_config(0).attn_config
    n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
    rope = self.config.build_rope(input_pos, n_elem, attn_config.rotary_base)

    if mask is None:
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, : self.config.kv_cache_max]

    return self._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, lora, export_config
    )

  def _forward_with_embeds(
      self,
      input_embeds: torch.Tensor,
      rope: Tuple[torch.Tensor, torch.Tensor],
      mask: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      lora: Optional[lora_utils.LoRA] = None,
      export_config: Optional[ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    """Forwards the model with input embeddings."""
    assert len(self.transformer_blocks) == len(kv_cache.caches), (
        "The number of transformer blocks and the number of KV cache entries"
        " must be the same."
    )

    x = input_embeds
    if self.config.embedding_scale is not None:
      x = x * self.config.embedding_scale

    updated_kv_entries = []
    for i, block in enumerate(self.transformer_blocks):
      kv_entry = kv_cache.caches[i] if kv_cache else None
      lora_adapter = lora.adapters[i] if lora else None
      x, kv_entry = block(x, rope, mask, input_pos, kv_entry, lora_adapter)
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
    logits = self.lm_head(x)  # (b, t, vocab_size)
    return {"logits": logits, "kv_cache": updated_kv_cache}


def build_decoder_only_model(
    checkpoint_path: str,
    config: cfg.ModelConfig,
    tensor_names: loading_utils.ModelLoader.TensorNames,
    model_class: type[nn.Module] = DecoderOnlyModel,
) -> nn.Module:
  transformer = model_class(config)
  loader = loading_utils.ModelLoader(checkpoint_path, tensor_names)
  loader.load(
      transformer, strict=not config.lm_head_share_weight_with_embedding
  )
  transformer.eval()
  return transformer
