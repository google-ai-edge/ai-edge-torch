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

import os
from pathlib import Path

import ai_edge_torch
# TODO @merron: Below import will be replaced by Qwen2
#from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.quantize import quant_recipes
import torch


def convert_qwen2_to_tflite(
    checkpoint_path: str,
    prefill_seq_len: int = 512,
    kv_cache_max_len: int = 1024,
    quantize: bool = True,
):
  """Converting a Qwen2 model to multi-signature
  tflite model.

  Args:
      checkpoint_path (str): The filepath to the model checkpoint, or directory holding the checkpoint.
      prefill_seq_len (int, optional): The maximum size of prefill input tensor.
        Defaults to 512.
      kv_cache_max_len (int, optional): The maximum size of KV cache buffer,
        including both prefill and decode. Defaults to 1024.
      quantize (bool, optional): Whether the model should be quanized.
        Defaults to True.
  """


if __name__ == '__main__':
  checkpoint_path = os.path.join(Path.home(), 'Downloads/llm_data/Qwen2')
  convert_qwen2_to_tflite(checkpoint_path)
