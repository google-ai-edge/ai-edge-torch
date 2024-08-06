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
#
# Note: This is an experimental version of Gemma with external KV cache.
# Please use with caution.


import os
from pathlib import Path

import ai_edge_torch
from ai_edge_torch.generative.examples.experimental.gemma import gemma
from ai_edge_torch.generative.layers.experimental import ekv_cache as kv_utils
from ai_edge_torch.generative.quantize import quant_recipes
import torch


def convert_gemma_to_tflite(
    checkpoint_path: str,
    prefill_seq_len: int = 512,
    kv_cache_max_len: int = 1024,
    quantize: bool = True,
):
  """An example method for converting a Gemma 2B model to multi-signature

  tflite model.
  Args:
      checkpoint_path (str): The filepath to the model checkpoint, or directory
        holding the checkpoint.
      prefill_seq_len (int, optional): The maximum size of prefill input tensor.
        Defaults to 512.
      kv_cache_max_len (int, optional): The maximum size of KV cache buffer,
        including both prefill and decode. Defaults to 1024.
      quantize (bool, optional): Whether the model should be quanized. Defaults
        to True.
  """
  pytorch_model = gemma.build_2b_model(
      checkpoint_path, kv_cache_max_len=kv_cache_max_len
  )
  # Tensors used to trace the model graph during conversion.
  prefill_tokens = torch.full((1, prefill_seq_len), 0, dtype=torch.long)
  prefill_input_pos = torch.arange(0, prefill_seq_len)
  decode_token = torch.tensor([[0]], dtype=torch.long)
  decode_input_pos = torch.tensor([0], dtype=torch.int64)
  kv = kv_utils.EKVCache.from_model_config(pytorch_model.config)

  quant_config = quant_recipes.full_int8_dynamic_recipe() if quantize else None
  edge_model = (
      ai_edge_torch.signature(
          'prefill',
          pytorch_model,
          sample_kwargs={
              'tokens': prefill_tokens,
              'input_pos': prefill_input_pos,
              'kv_cache': kv,
          },
      )
      .signature(
          'decode',
          pytorch_model,
          sample_kwargs={
              'tokens': decode_token,
              'input_pos': decode_input_pos,
              'kv_cache': kv,
          },
      )
      .convert(quant_config=quant_config)
  )
  edge_model.export(
      f'/tmp/gemma_seq{prefill_seq_len}_ekv{kv_cache_max_len}.tflite'
  )


if __name__ == '__main__':
  checkpoint_path = os.path.join(Path.home(), 'Downloads/llm_data/gemma-2b')
  convert_gemma_to_tflite(checkpoint_path)
