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

"""Example of converting a Gemma3 model to multi-signature tflite model."""

import os
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers.experimental import kv_cache
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.model_builder import ExportConfig
import torch

_MODEL_SIZE = flags.DEFINE_string(
    'model_size',
    '1b',
    'The size of the model to convert.',
)

_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    os.path.join(pathlib.Path.home(), 'Downloads/llm_data/gemma3-1b'),
    'The path to the model checkpoint, or directory holding the checkpoint.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    '/tmp/',
    'The path to export the tflite model.',
)
_OUTPUT_NAME_PREFIX = flags.DEFINE_string(
    'output_name_prefix',
    'gemma3',
    'The prefix of the output tflite model name.',
)
_PREFILL_SEQ_LENS = flags.DEFINE_multi_integer(
    'prefill_seq_lens',
    (32, 64, 128, 256, 512, 1024),
    'List of the maximum sizes of prefill input tensors.',
)
_KV_CACHE_MAX_LEN = flags.DEFINE_integer(
    'kv_cache_max_len',
    2048,
    'The maximum size of KV cache buffer, including both prefill and decode.',
)
_QUANTIZE = flags.DEFINE_bool(
    'quantize',
    False,
    'Whether the model should be quantized.',
)
_LORA_RANKS = flags.DEFINE_multi_integer(
    'lora_ranks',
    None,
    'If set, the model will be converted with the provided list of LoRA ranks.',
)


def _create_mask(mask_len, kv_cache_max_len):
  mask = torch.full(
      (mask_len, kv_cache_max_len), float('-inf'), dtype=torch.float32
  )
  mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
  return mask


def _create_export_config(
    prefill_seq_lens: list[int], kv_cache_max_len: int
) -> ExportConfig:
  """Creates the export config for the model."""
  export_config = ExportConfig()
  if isinstance(prefill_seq_lens, list):
    prefill_mask = [_create_mask(i, kv_cache_max_len) for i in prefill_seq_lens]
  else:
    prefill_mask = _create_mask(prefill_seq_lens, kv_cache_max_len)

  export_config.prefill_mask = prefill_mask

  decode_mask = torch.full(
      (1, kv_cache_max_len), float('-inf'), dtype=torch.float32
  )
  decode_mask = torch.triu(decode_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
  export_config.decode_mask = decode_mask
  export_config.kvcache_cls = kv_cache.KVCacheTransposed
  return export_config


def main(_):
  if _MODEL_SIZE.value == '1b':
    pytorch_model = gemma3.build_model_1b(
        _CHECKPOINT_PATH.value, kv_cache_max_len=_KV_CACHE_MAX_LEN.value
    )
    config = pytorch_model.config
  else:
    raise ValueError(f'Unsupported model size: {_MODEL_SIZE.value}')
  converter.convert_to_tflite(
      pytorch_model,
      output_path=_OUTPUT_PATH.value,
      output_name_prefix=_OUTPUT_NAME_PREFIX.value,
      prefill_seq_len=_PREFILL_SEQ_LENS.value,
      quantize=_QUANTIZE.value,
      config=config,
      lora_ranks=_LORA_RANKS.value,
      export_config=_create_export_config(
          _PREFILL_SEQ_LENS.value, _KV_CACHE_MAX_LEN.value
      ),
  )


if __name__ == '__main__':
  app.run(main)
