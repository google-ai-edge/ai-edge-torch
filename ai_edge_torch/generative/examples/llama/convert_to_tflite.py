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

"""Example of converting Llama 3.2 1B model to multi-signature tflite model."""

import os
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.llama import llama
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.model_builder import ExportConfig

_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '1b',
    ['1b', '3b'],
    'The size of the model to verify.',
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    os.path.join(pathlib.Path.home(), 'Downloads/llm_data/llama'),
    'The path to the model checkpoint, or directory holding the checkpoint.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    '/tmp/',
    'The path to export the tflite model.',
)
_OUTPUT_NAME_PREFIX = flags.DEFINE_string(
    'output_name_prefix',
    'llama',
    'The prefix of the output tflite model name.',
)
_PREFILL_SEQ_LENS = flags.DEFINE_multi_integer(
    'prefill_seq_lens',
    (8, 64, 128, 256, 512, 1024),
    'List of the maximum sizes of prefill input tensors.',
)
_KV_CACHE_MAX_LEN = flags.DEFINE_integer(
    'kv_cache_max_len',
    1280,
    'The maximum size of KV cache buffer, including both prefill and decode.',
)
_QUANTIZE = flags.DEFINE_bool(
    'quantize',
    True,
    'Whether the model should be quantized.',
)
_LORA_RANKS = flags.DEFINE_multi_integer(
    'lora_ranks',
    None,
    'If set, the model will be converted with the provided list of LoRA ranks.',
)

_BUILDER = {
    '1b': llama.build_1b_model,
    '3b': llama.build_3b_model,
}


def main(_):
  pytorch_model = _BUILDER[_MODEL_SIZE.value](
      _CHECKPOINT_PATH.value, kv_cache_max_len=_KV_CACHE_MAX_LEN.value
  )
  converter.convert_to_tflite(
      pytorch_model,
      output_path=_OUTPUT_PATH.value,
      output_name_prefix=_OUTPUT_NAME_PREFIX.value,
      prefill_seq_len=_PREFILL_SEQ_LENS.value,
      quantize=_QUANTIZE.value,
      lora_ranks=_LORA_RANKS.value,
      export_config=ExportConfig(),
  )


if __name__ == '__main__':
  app.run(main)
