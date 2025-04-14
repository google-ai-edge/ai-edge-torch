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

from absl import app
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
import torch

flags = converter.define_conversion_flags('gemma3-1b')
ExportConfig = export_config.ExportConfig


_MODEL_SIZE = flags.DEFINE_string(
    'model_size',
    '1b',
    'The size of the model to convert.',
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
  export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
  return export_config


def main(_):
  if _MODEL_SIZE.value == '1b':
    pytorch_model = gemma3.build_model_1b(
        flags.FLAGS.checkpoint_path,
        kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
    )
    config = pytorch_model.config
  else:
    raise ValueError(f'Unsupported model size: {_MODEL_SIZE.value}')
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=flags.FLAGS.output_name_prefix,
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      quantize=flags.FLAGS.quantize,
      config=config,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=_create_export_config(
          flags.FLAGS.prefill_seq_lens, flags.FLAGS.kv_cache_max_len
      ),
  )


if __name__ == '__main__':
  app.run(main)
