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

"""Example of building a SmolLM model."""

import copy
import os
import pathlib

from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import numpy as np
import torch
from torch import nn

TENSOR_NAMES = copy.copy(tiny_llama.TENSOR_NAMES)
# SmolLM re-uses the embedding as the head projection layer.
TENSOR_NAMES.lm_head = None


class SmolLM(tiny_llama.TinyLlama):
  """A SmolLM model built from the Edge Generative API layers.

  SmolLM shares the same architecture as TinyLlama, but with different model
  sizes.
  """

  def __init__(self, config: cfg.ModelConfig):
    super().__init__(config)
    # SmolLM re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data


def get_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:
  """Returns the model config for a SmolLM 135M model.

  Args:
    kv_cache_max_len (int): The maximum sequence length of the KV cache. Default
      is 1024.

  Returns:
    The model config for a SmolLM model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=9,
      head_dim=64,
      num_query_groups=3,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=1536,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=49152,
      num_layers=30,
      max_seq_len=2048,
      embedding_dim=576,
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
  # SmolLM has only one block config.
  config.block_config(0).ff_config.intermediate_size = 64
  return config


def build_model(checkpoint_path: str, **kwargs) -> nn.Module:
  config = get_model_config(**kwargs)
  model = SmolLM(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # Since embedding and lm-head use the same weight, we need to set strict
  # to False.
  loader.load(model, strict=False)
  model.eval()
  return model


def define_and_run(checkpoint_path: str) -> None:
  """Instantiates and runs a SmolLM model."""

  current_dir = pathlib.Path(__file__).parent.resolve()
  smollm_goldens = torch.load(current_dir / "smollm_lm_logits.pt")
  kv_cache_max_len = 1024
  model = build_model(checkpoint_path, kv_cache_max_len=kv_cache_max_len)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, kv_cache_max_len), 0, dtype=torch.int, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, kv_cache_max_len, dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(model.config)
  output = model.forward(tokens, input_pos, kv)
  assert torch.allclose(
      smollm_goldens, output["logits"][0, idx.shape[1] - 1, :], atol=1e-05
  )


if __name__ == "__main__":
  input_checkpoint_path = os.path.join(
      pathlib.Path.home(), "Downloads/llm_data/smollm"
  )
  define_and_run(input_checkpoint_path)
