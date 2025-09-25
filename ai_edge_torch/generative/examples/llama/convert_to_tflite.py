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

from absl import app
from ai_edge_torch.generative.examples.llama import llama
from ai_edge_torch.generative.utilities import converter

flags = converter.define_conversion_flags('llama')

_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '1b',
    ['1b', '3b'],
    'The size of the model to verify.',
)

_BUILDER = {
    '1b': llama.build_1b_model,
    '3b': llama.build_3b_model,
}


def main(_):
  converter.build_and_convert_to_tflite_from_flags(_BUILDER[_MODEL_SIZE.value])


if __name__ == '__main__':
  app.run(main)
