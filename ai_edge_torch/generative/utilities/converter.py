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

"""Common utility functions for model conversion."""

from typing import Union

from ai_edge_torch._convert import converter as converter_utils
import ai_edge_torch.generative.layers.kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.quantize import quant_recipes
import torch


def convert_to_tflite(
    pytorch_model: torch.nn.Module,
    tflite_path: str,
    prefill_seq_len: Union[int, list[int]],
    pixel_values_size: torch.Size = None,
    quantize: bool = True,
    config: cfg.ModelConfig = None,
):
  """Converts a nn.Module model to multi-signature tflite model.

  A PyTorch model will be converted to a tflite model with several signatures:
    * "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
        passed),
    * "prefill_[preill_seq_len]_pixel" (or "prefill_pixel" if only one
        prefill_seq_len is passed) if num_pixel_values > 0, and
    * "decode".

  "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
  passed) signature takes as a sample input:
    * a tensor of shape [1, prefill_seq_len] of token sequence,
    * a tensor of shape [1, prefill_seq_len] of token positions, and
    * an external KV cache.

  If num_pixel_values > 0, "prefill_[prefill_seq_len]_pixel" (or "prefill_pixel"
  if only one prefill_seq_len is passed) signature takes as a sample input:
    * a tensor of shape [1, prefill_seq_len] of token sequence,
    * a tensor of shape [1, prefill_seq_len] of token positions,
    * an external KV cache, and
    * a tensor of shape [1, num_pixel_values] of pixel values.

  "decode" signature takes as a sample input:
    * a tensor of shape [1, 1] of token sequence,
    * a tensor of shape [1, 1] of the token position, and
    * an external KV cache.

  The final tflite model will be exported to tflite_path.

  Args:
      pytorch_model (torch.nn.Module): PyTorch model to convert to tflite.
      tflite_path (str): The tflite file path to export.
      prefill_seq_len (Union[int, list[int]]): A list of prefill lengths to
        export.
      pixel_values_size (torch.Size, optional): The size of pixel values to pass
        to the model. If None, the model is not expected to take pixel values.
      quantize (bool, optional): Whether the model should be quanized. Defaults
        to True.
      config (cfg.ModelConfig, optional): The model config used to configure KV
        cache. If None, it uses the config of the pytorch_model.
  """
  prefill_seq_lens = (
      [prefill_seq_len] if isinstance(prefill_seq_len, int) else prefill_seq_len
  )

  # Tensors used to trace the model graph during conversion.
  prefill_tokens_list = []
  prefill_input_pos_list = []
  for seq_len in prefill_seq_lens:
    prefill_tokens_list.append(torch.full((1, seq_len), 0, dtype=torch.int))
    prefill_input_pos_list.append(torch.arange(0, seq_len, dtype=torch.int))

  prefill_pixel_values = (
      torch.full((1,) + pixel_values_size, 0, dtype=torch.float32)
      if pixel_values_size
      else None
  )

  decode_token = torch.tensor([[0]], dtype=torch.int)
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(
      config if config else pytorch_model.config
  )

  quant_config = quant_recipes.full_int8_dynamic_recipe() if quantize else None
  converter = converter_utils.Converter()
  for i in range(len(prefill_seq_lens)):
    prefill_seq_len = prefill_seq_lens[i]
    prefill_tokens = prefill_tokens_list[i]
    prefill_input_pos = prefill_input_pos_list[i]
    if i == 0 and len(prefill_seq_lens) == 1:
      prefill_signature_name = 'prefill'
    else:
      prefill_signature_name = f'prefill_{prefill_seq_len}'
    converter.add_signature(
        prefill_signature_name,
        pytorch_model,
        sample_kwargs={
            'tokens': prefill_tokens,
            'input_pos': prefill_input_pos,
            'kv_cache': kv,
        },
    )
    if prefill_pixel_values is not None:
      converter.add_signature(
          prefill_signature_name + '_pixel',
          pytorch_model,
          sample_kwargs={
              'tokens': prefill_tokens,
              'input_pos': prefill_input_pos,
              'kv_cache': kv,
              'pixel_values': prefill_pixel_values,
          },
      )

  converter.add_signature(
      'decode',
      pytorch_model,
      sample_kwargs={
          'tokens': decode_token,
          'input_pos': decode_input_pos,
          'kv_cache': kv,
      },
  )

  edge_model = converter.convert(quant_config=quant_config)
  edge_model.export(tflite_path)
