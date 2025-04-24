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

"""Example of converting Qwen 2.5 models to multi-signature tflite model."""

from absl import app
from ai_edge_torch.generative.examples.qwen import qwen
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
import torch

flags = converter.define_conversion_flags('qwen')
ExportConfig = export_config.ExportConfig

_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '3b',
    ['0.5b', '1.5b', '3b'],
    'The size of the model to convert.',
)

_BUILDER = {
    '0.5b': qwen.build_0_5b_model,
    '1.5b': qwen.build_1_5b_model,
    '3b': qwen.build_3b_model,
}


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
  pytorch_model = _BUILDER[_MODEL_SIZE.value](
      flags.FLAGS.checkpoint_path, kv_cache_max_len=flags.FLAGS.kv_cache_max_len
  )
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=flags.FLAGS.output_name_prefix,
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      quantize=flags.FLAGS.quantize,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=_create_export_config(
          flags.FLAGS.prefill_seq_lens, flags.FLAGS.kv_cache_max_len
      )
      if flags.FLAGS.transpose_kv_cache
      else ExportConfig(),
  )


if __name__ == '__main__':
  app.run(main)
