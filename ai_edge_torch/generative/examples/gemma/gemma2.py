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
# Example of building the Gemma2 2B model.

import os
from pathlib import Path
from typing import Optional, Tuple

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
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


class Gemma2Block(attention.TransformerBlock):

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward function of the Gemma2Block.

    Exactly the same as TransformerBlock but we call the post-attention norm
    immediately after attention and not after the residual pointwise addition.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.

    Returns:
      output activation from this transformer block.
    """

    x_norm = self.pre_atten_norm(x)
    attn_out = self.atten_func(x_norm, rope, mask, input_pos)
    attn_out_norm = self.post_atten_norm(attn_out)
    x = x + attn_out_norm
    output = x + self.ff(x)
    return output


class Gemma2(nn.Module):
  """A Gemma2 model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    self.config = config
    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim,
        config.vocab_size,
        bias=config.lm_head_use_bias,
    )
    # Gemma re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        Gemma2Block(config) for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(
            config.attn_config.rotary_percentage * config.attn_config.head_dim
        ),
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

    self.sliding_window_mask_cache = attn_utils.build_sliding_window_mask_cache(
        size=config.kv_cache_max,
        window_size=self.config.attn_config.sliding_window_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    self.config = config

  def get_attention_mask(
      self, idx: int, input_pos: torch.Tensor
  ) -> torch.Tensor:
    if self.config.attn_config.attn_types:
      if (
          self.config.attn_config.attn_types[idx]
          == cfg.AttentionType.LOCAL_SLIDING
      ):
        return self.sliding_window_mask_cache.index_select(2, input_pos)

    return self.mask_cache.index_select(2, input_pos)

  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    _, seq_len = idx.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)

    # token embeddings of shape (b, t, n_embd)
    x = self.tok_embedding(idx)
    x = x * (self.config.embedding_dim**0.5)

    for i, block in enumerate(self.transformer_blocks):
      mask = self.get_attention_mask(i, input_pos)
      x = block(x, (cos, sin), mask, input_pos)

    x = self.final_norm(x)
    res = self.lm_head(x)  # (b, t, vocab_size)
    if self.config.final_logit_softcap is not None:
      res = res / self.config.final_logit_softcap
      res = torch.tanh(res)
      res = res * self.config.final_logit_softcap
    return res


def get_model_config_2b(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a Gemma2 2B model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a Gemma 2B model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=8,
      head_dim=256,
      num_query_groups=4,
      rotary_percentage=1.0,
      qkv_transpose_before_split=True,
      logit_softcap=50.0,
      sliding_window_size=4096,
      attn_types=[cfg.AttentionType.GLOBAL, cfg.AttentionType.LOCAL_SLIDING]
      * 13,
  )

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
  config = cfg.ModelConfig(
      vocab_size=256000,
      num_layers=26,
      max_seq_len=8192,
      embedding_dim=2304,
      kv_cache_max_len=kv_cache_max_len,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      final_norm_config=norm_config,
      parallel_residual=False,
      lm_head_use_bias=False,
      enable_hlfb=True,
      final_logit_softcap=30.0,
  )
  return config


def get_fake_model_config(kv_cache_max_len: int = 128) -> cfg.ModelConfig:
  config = get_model_config_2b(kv_cache_max_len)
  config.attn_config.num_heads = 4
  config.attn_config.head_dim = 64
  config.attn_config.sliding_window_size = 64
  config.ff_config.intermediate_size = 128
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 2 * kv_cache_max_len
  config.embedding_dim = 128
  return config


def build_2b_model(checkpoint_path: str, **kwargs) -> nn.Module:
  config = get_model_config_2b(**kwargs)
  model = Gemma2(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # since embedding and lm-head use the same weight, we need to set strict
  # to False.
  loader.load(model, strict=False)
  model.eval()
  return model


def define_and_run_2b() -> None:
  """Instantiates and runs a Gemma2 2B model."""

  current_dir = Path(__file__).parent.resolve()
  gemma2_goldens = torch.load(current_dir / "gemma2it_2b_golden.pt")
  print("Running GEMMA 2")
  kv_cache_max_len = 1024
  checkpoint_path = os.path.join(Path.home(), "Downloads/llm_data/gemma2-2b")
  model = build_2b_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  toks = torch.from_numpy(
      np.array([2, 651, 9456, 576, 573, 3520, 3858, 603, 235248])
  )
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.long, device="cpu")
  tokens[0, :9] = toks
  input_pos = torch.arange(0, kv_cache_max_len)
  out = model.forward(tokens, input_pos)
  out_final = out[0, 8, :]
  assert torch.allclose(gemma2_goldens, out_final, atol=1e-04)


if __name__ == "__main__":
  torch.set_printoptions(sci_mode=True)
  define_and_run_2b()
