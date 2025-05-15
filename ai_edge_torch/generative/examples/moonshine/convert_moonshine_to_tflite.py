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

"""Example of converting a Moonshine model to multi-signature tflite model."""

import os
import pathlib

from absl import app
from absl import flags
import ai_edge_torch
from ai_edge_torch.generative.examples.moonshine import moonshine
import torch

_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    os.path.join(pathlib.Path.home(), 'Downloads/llm_data/moonshine'),
    'The path to the model checkpoint, or directory holding the checkpoint.',
)
_TFLITE_PATH = flags.DEFINE_string(
    'tflite_path',
    '/tmp/',
    'The tflite file path to export.',
)


def main(_):
  p_model = moonshine.build_preprocessor(_CHECKPOINT_PATH.value)
  output_filename = f'moonshine_preprocessor.tflite'
  _input = torch.randn((1, 1, 159414), dtype=torch.float)
  edge_model = ai_edge_torch.convert(p_model, (_input,), quant_config=None)
  tflite_path = os.path.join(_TFLITE_PATH.value, output_filename)
  edge_model.export(tflite_path)


if __name__ == '__main__':
  app.run(main)
