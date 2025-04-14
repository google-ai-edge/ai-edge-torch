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

"""Example of building a decoder of PaliGemma2 3B model which is Gemma2."""

from typing import Optional

from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import export_config as export_cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="language_model.model.layers.{}.mlp.up_proj",
    ff_down_proj="language_model.model.layers.{}.mlp.down_proj",
    ff_gate_proj="language_model.model.layers.{}.mlp.gate_proj",
    attn_query_proj="language_model.model.layers.{}.self_attn.q_proj",
    attn_key_proj="language_model.model.layers.{}.self_attn.k_proj",
    attn_value_proj="language_model.model.layers.{}.self_attn.v_proj",
    attn_output_proj="language_model.model.layers.{}.self_attn.o_proj",
    pre_attn_norm="language_model.model.layers.{}.input_layernorm",
    post_attn_norm="language_model.model.layers.{}.post_attention_layernorm",
    pre_ff_norm="language_model.model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="language_model.model.layers.{}.post_feedforward_layernorm",
    embedding="language_model.model.embed_tokens",
    final_norm="language_model.model.norm",
    lm_head=None,
)


class Decoder2(gemma2.Gemma2):
  """A decoder of PaliGemma2 3B model which is Gemma2.

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
      mask: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if input_embeds is None:
      return super().forward(tokens, input_pos, kv_cache, mask, export_config)

    assert input_embeds is not None

    rope_pos = input_pos + 1  # PaliGemma2 position is 1-based.
    # ROPE parameters for all attn_configs are the same. Take the first one.
    attn_config = self.config.block_config(0).attn_config
    n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
    rope = self.config.build_rope(rope_pos, n_elem, attn_config.rotary_base)

    if mask is None:
      # By default, don't mask image embeds with a diagonal causal mask.
      embeds_len = input_embeds.shape[1]
      mask = torch.zeros(embeds_len, self.config.kv_cache_max)
      mask[:, embeds_len:] = float("-inf")

    return self._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, export_config
    )


def get_decoder2_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for the decoder of a PaliGemma 3B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for the decoder of a PaliGemma 3B model.
  """
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=True,
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
        logit_softcap=50.0,
        sliding_window_size=4096,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if idx % 2 == 0
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
      vocab_size=257216,
      num_layers=num_layers,
      max_seq_len=8192,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      enable_hlfb=True,
      final_logit_softcap=30.0,
  )
  return config


def get_fake_decoder2_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_decoder2_config(kv_cache_max_len)
  # PaliGemma2 decoder has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  config.embedding_dim = 128
  config.embedding_scale = 128**0.5
  return config


def build_decoder2(checkpoint_path: str, **kwargs) -> torch.nn.Module:
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_decoder2_config(**kwargs),
      tensor_names=TENSOR_NAMES,
      model_class=Decoder2,
  )
