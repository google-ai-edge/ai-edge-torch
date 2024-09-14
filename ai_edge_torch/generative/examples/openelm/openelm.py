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

"""Example of building an OpenELM model."""

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
    ff_up_proj="transformer.layers.{}.ffn.proj_1",
    ff_down_proj="transformer.layers.{}.ffn.proj_2",
    attn_fused_qkv_proj="transformer.layers.{}.attn.qkv_proj",
    attn_query_norm="transformer.layers.{}.attn.q_norm",
    attn_key_norm="transformer.layers.{}.attn.k_norm",
    attn_output_proj="transformer.layers.{}.attn.out_proj",
    pre_attn_norm="transformer.layers.{}.attn_norm",
    pre_ff_norm="transformer.layers.{}.ffn_norm",
    embedding="transformer.token_embeddings",
    final_norm="transformer.norm",
    lm_head=None,
)


class OpenELM(nn.Module):
  """An OpenELM model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    # OpenELM re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    # OpenELM has same hyper parameters for rotary_percentage and head_dim for
    # each layer block. Use the first block.
    attn_config = config.block_config(0).attn_config
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
  """Returns the model config for an OpenELM model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for an OpenELM model.
  """
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6
  )
  num_heads = [12] * 4 + [16] * 14 + [20] * 12 + [24] * 6
  num_query_groups = [3] * 4 + [4] * 14 + [5] * 12 + [6] * 6

  def make_divisible(v, d):
    """Ensures that all layers have a channel number that is divisible by d."""
    new_v = int(v + d / 2) // d * d
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
      new_v += d
    return new_v

  # The way to get intermediate size is from
  # https://huggingface.co/apple/OpenELM-3B/blob/main/modeling_openelm.py
  def get_intermediate_size(idx: int) -> int:
    return make_divisible((0.5 + 0.1 * idx) * 3072, 256)

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    return cfg.TransformerBlockConfig(
        attn_config=cfg.AttentionConfig(
            num_heads=num_heads[idx],
            head_dim=128,
            num_query_groups=num_query_groups[idx],
            rotary_percentage=1.0,
            qkv_transpose_before_split=True,
            query_norm_config=norm_config,
            key_norm_config=norm_config,
        ),
        ff_config=cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=cfg.ActivationConfig(
                cfg.ActivationType.SILU_GLU, gate_is_front=True
            ),
            intermediate_size=get_intermediate_size(idx),
            pre_ff_norm_config=norm_config,
        ),
        pre_attention_norm_config=norm_config,
    )

  num_layers = 36
  config = cfg.ModelConfig(
      vocab_size=32000,
      num_layers=num_layers,
      max_seq_len=2048,
      embedding_dim=3072,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
  )
  return config


def get_fake_model_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_model_config(kv_cache_max_len)
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  config.embedding_dim = 128
  config.block_configs = config.block_configs[: config.num_layers]
  for block_config in config.block_configs:
    block_config.attn_config.num_heads = 3
    block_config.attn_config.head_dim = 64
    block_config.ff_config.intermediate_size = 128
  return config


def build_model(checkpoint_path: str, **kwargs) -> nn.Module:
  config = get_model_config(**kwargs)
  model = OpenELM(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # Since embedding and lm-head use the same weight, we need to set strict
  # to False.
  loader.load(model, strict=False)
  model.eval()
  return model


def define_and_run(checkpoint_path: str) -> None:
  """Instantiates and runs an OpenELM model."""

  current_dir = pathlib.Path(__file__).parent.resolve()
  openelm_goldens = torch.load(current_dir / "openelm_lm_logits.pt")
  kv_cache_max_len = 1024
  model = build_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.int, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len, dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(model.config)
  output = model.forward(tokens, input_pos, kv)
  assert torch.allclose(
      openelm_goldens, output["logits"][0, idx.shape[1] - 1, :], atol=1e-05
  )


if __name__ == "__main__":
  input_checkpoint_path = os.path.join(
      pathlib.Path.home(), "Downloads/llm_data/openelm"
  )
  define_and_run(input_checkpoint_path)
