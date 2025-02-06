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

"""Example of building decoder for Qwen 2.5 VL models."""

from typing import Optional, Tuple

from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import torch

TENSOR_NAMES = model_builder.TENSOR_NAMES


class Decoder(model_builder.DecoderOnlyModel):
  """A decoder for Qwen-VL model built from the Edge Generative API layers.

  Besides a tensor of text token IDs, forward() can also take a tensor of
  embeddings which may include text or image or both.
  """

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      input_embeds: torch.Tensor = None,
      rope: Tuple[torch.Tensor, torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
      export_config: Optional[model_builder.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if input_embeds is None:
      _, seq_len = tokens.size()
      assert self.config.max_seq_len >= seq_len, (
          f"Cannot forward sequence of length {seq_len}, max seq length is only"
          f" {self.config.max_seq_len}"
      )
      # token embeddings of shape (b, t, n_embd)
      input_embeds = self.tok_embedding(tokens)

    if rope is None:
      # ROPE parameters for all attn_configs are the same. Take the first one.
      attn_config = self.config.block_config(0).attn_config
      n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
      rope = self.config.build_rope(input_pos, n_elem, attn_config.rotary_base)

    if mask is None:
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, : self.config.kv_cache_max]

    return self._forward_with_embeds(
        input_embeds,
        rope,
        mask,
        input_pos,
        kv_cache,
        export_config=export_config,
    )


def get_decoder_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 VL 3B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a Qwen 2.5 VL 3B model.
  """
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
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-06,
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
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_fake_decoder_config(**kwargs) -> cfg.ModelConfig:
  config = get_decoder_config(**kwargs)
  config.vocab_size = 128
  config.num_layers = 2
  # Decoder has only one block config.
  config.block_config(0).ff_config.intermediate_size = 64
  return config


def build_decoder(checkpoint_path: str, **kwargs) -> torch.nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_decoder_config(**kwargs),
      tensor_names=TENSOR_NAMES,
      model_class=Decoder,
  )
