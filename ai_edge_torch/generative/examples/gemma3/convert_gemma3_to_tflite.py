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
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
from ai_edge_torch.generative.utilities import loader

flags = converter.define_conversion_flags(
    'gemma3-1b', default_mask_as_input=True, default_transpose_kv_cache=True
)

_CUSTOM_CHECKPOINT_LOADER = flags.DEFINE_bool(
    'custom_checkpoint_loader',
    False,
    'If true, the conversion script will use a custom checkpoint loader which'
    ' will read a checkpoint from a remote source.',
)

_MODEL_SIZE = flags.DEFINE_string(
    'model_size',
    '1b',
    'The size of the model to convert.',
)


def main(_):
  custom_loader = None
  if flags.FLAGS.custom_checkpoint_loader:
    # If loading from a remote source, try to get a custom loader first.
    custom_loader = loader.get_custom_loader(flags.FLAGS.checkpoint_path)

  if _MODEL_SIZE.value == '1b':
    pytorch_model = gemma3.build_model_1b(
        flags.FLAGS.checkpoint_path,
        kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
        custom_loader=custom_loader,
    )
  else:
    raise ValueError(f'Unsupported model size: {_MODEL_SIZE.value}')

  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=flags.FLAGS.output_name_prefix,
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      quantize=flags.FLAGS.quantize,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=export_config.get_from_flags(),
  )


if __name__ == '__main__':
  app.run(main)
