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
# A toy example which has basic transformer block (w/ KV-Cache).
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_xla

import ai_edge_torch
from ai_edge_torch.generative.layers.attention import TransformerBlock
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.builder as builder
import ai_edge_torch.generative.layers.model_config as cfg

RoPECache = Tuple[torch.Tensor, torch.Tensor]


class ToyModelWithKV(torch.nn.Module):

  def __init__(self, config: cfg.ModelConfig) -> None:
    super().__init__()
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    self.tok_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
    self.transformer_blocks = nn.ModuleList(
        TransformerBlock(config) for _ in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.max_seq_len,
        dim=int(config.attn_config.rotary_percentage * config.head_dim),
        base=10_000,
        condense_ratio=1,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.max_seq_len, dtype=torch.float32, device=torch.device('cpu')
    )
    self.config = config

  @torch.inference_mode
  def forward(self, idx: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    x = self.tok_embedding(idx)
    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, : self.config.max_seq_len]
    for i, block in enumerate(self.transformer_blocks):
      x = block(x, (cos, sin), mask, input_pos)
    x = self.final_norm(x)
    return self.lm_head(x)


def _export_stablehlo_mlir(model, args):
  ep = torch.export.export(model, args)
  stablehlo_gm = torch_xla.stablehlo.exported_program_to_stablehlo(ep)
  return stablehlo_gm.get_stablehlo_text()


def get_model_config() -> cfg.ModelConfig:
  attn_config = cfg.AttentionConfig(
      num_heads=32, num_query_groups=4, rotary_percentage=1.0
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=256,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  config = cfg.ModelConfig(
      vocab_size=150,
      num_layers=2,
      max_seq_len=500,
      embedding_dim=128,
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      pre_ff_norm_config=norm_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config


def get_sample_prefill_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
  idx = torch.unsqueeze(torch.arange(0, 100), 0)
  input_pos = torch.arange(0, 100)
  return idx, input_pos


def get_sample_decode_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
  idx = torch.tensor([[1]], dtype=torch.long)
  input_pos = torch.tensor([10], dtype=torch.int64)
  return idx, input_pos


def define_and_run() -> None:
  dump_mlir = False

  config = get_model_config()
  model = ToyModelWithKV(config)
  print('running an inference')
  idx, input_pos = get_sample_prefill_inputs()
  decode_idx, decode_input_pos = get_sample_decode_inputs()
  print(model.forward(idx, input_pos))

  if dump_mlir:
    mlir_text = _export_stablehlo_mlir(model, (idx, input_pos))
    with open('/tmp/toy_model_with_kv.stablehlo.mlir', 'w') as f:
      f.write(mlir_text)

  # Convert model to tflite with 2 signatures (prefill + decode).
  print('converting toy model to tflite with 2 signatures (prefill + decode)')
  edge_model = (
      ai_edge_torch.signature('prefill', model, (idx, input_pos))
      .signature('decode', model, (decode_idx, decode_input_pos))
      .convert()
  )
  edge_model.export('/tmp/toy_kv_cache.tflite')


if __name__ == '__main__':
  define_and_run()
