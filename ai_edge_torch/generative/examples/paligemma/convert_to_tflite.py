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

"""Example of converting a PaliGemma model to multi-signature tflite model."""

import os
import pathlib

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.paligemma import paligemma
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.model_builder import ExportConfig
import torch

_VERSION = flags.DEFINE_enum(
    'version',
    '2',
    ['1', '2'],
    'The version of PaliGemma model to verify.',
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    os.path.join(pathlib.Path.home(), 'Downloads/llm_data/paligemma2-3b-224'),
    'The path to the model checkpoint, or directory holding the checkpoint.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    '/tmp/',
    'The path to export the tflite model.',
)
_OUTPUT_NAME_PREFIX = flags.DEFINE_string(
    'output_name_prefix',
    'paligemma',
    'The prefix of the output tflite model name.',
)
_PREFILL_SEQ_LEN = flags.DEFINE_integer(
    'prefill_seq_len',
    1024,
    'The maximum size of prefill input tensor.',
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


def main(_):
  pytorch_model = paligemma.build_model(
      _CHECKPOINT_PATH.value,
      version=int(_VERSION.value),
      kv_cache_max_len=_KV_CACHE_MAX_LEN.value,
  )

  config = pytorch_model.image_encoder.config.image_embedding
  converter.convert_to_tflite(
      pytorch_model,
      output_path=_OUTPUT_PATH.value,
      output_name_prefix=f'{_OUTPUT_NAME_PREFIX.value}_{_VERSION.value}',
      prefill_seq_len=_PREFILL_SEQ_LEN.value,
      pixel_values_size=torch.Size(
          [1, config.channels, config.image_size, config.image_size]
      ),
      quantize=_QUANTIZE.value,
      config=pytorch_model.config.decoder_config,
      export_config=ExportConfig(),
  )


if __name__ == '__main__':
  app.run(main)
