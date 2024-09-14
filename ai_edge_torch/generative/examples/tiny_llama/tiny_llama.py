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

"""Example of building a TinyLlama model."""

import os
import pathlib

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import numpy as np
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
    lm_head="lm_head",
)


class TinyLlama(nn.Module):
  """A TinyLlama model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    # Construct model layers.
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    # TinyLlama has only one block config.
    block_config = config.block_config(0)
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(block_config, config)
        for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    attn_config = block_config.attn_config
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=10_000,
        condense_ratio=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )
    assert len(self.transformer_blocks) == len(kv_cache.caches), (
        "The number of transformer blocks and the number of KV cache entries"
        " must be the same."
    )

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.kv_cache_max]

    # token embeddings of shape (b, t, n_embd)
    x = self.tok_embedding(tokens)

    updated_kv_entires = []
    for i, block in enumerate(self.transformer_blocks):
      kv_entry = kv_cache.caches[i] if kv_cache else None
      x, kv_entry = block(x, (cos, sin), mask, input_pos, kv_entry)
      if kv_entry:
        updated_kv_entires.append(kv_entry)
    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entires))

    x = self.final_norm(x)
    logits = self.lm_head(x)  # (b, t, vocab_size)
    return {"logits": logits, "kv_cache": updated_kv_cache}


def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a TinyLlama model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a TinyLlama model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=32,
      head_dim=64,
      num_query_groups=4,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=5632,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=32000,
      num_layers=22,
      max_seq_len=2048,
      embedding_dim=2048,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_fake_model_config(**kwargs) -> cfg.ModelConfig:
  config = get_model_config(**kwargs)
  config.vocab_size = 128
  config.num_layers = 2
  # TinyLlama has only one block config.
  config.block_config(0).ff_config.intermediate_size = 64
  return config


def build_model(checkpoint_path: str, **kwargs) -> nn.Module:
  config = get_model_config(**kwargs)
  model = TinyLlama(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(model)
  model.eval()
  return model


def define_and_run(checkpoint_path: str) -> None:
  """Instantiates and runs a TinyLlama model."""

  current_dir = pathlib.Path(__file__).parent.resolve()
  tiny_llama_goldens = torch.load(current_dir / "tiny_llama_lm_logits.pt")
  kv_cache_max_len = 1024
  model = build_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.int, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len, dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(model.config)
  output = model.forward(tokens, input_pos, kv)
  assert torch.allclose(
      tiny_llama_goldens, output["logits"][0, idx.shape[1] - 1, :], atol=1e-02
  )


if __name__ == "__main__":
  input_checkpoint_path = os.path.join(
      pathlib.Path.home(), "Downloads/llm_data/tiny_llama"
  )
  define_and_run(input_checkpoint_path)
