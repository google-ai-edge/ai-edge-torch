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

from absl import app
from ai_edge_torch.generative.examples.paligemma import paligemma
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
from ai_edge_torch.generative.utilities import loader
import torch

flags = converter.define_conversion_flags('paligemma2-3b-224')

_VERSION = flags.DEFINE_enum(
    'version',
    '2',
    ['1', '2'],
    'The version of PaliGemma model to verify.',
)


def main(_):
  checkpoint_path = flags.FLAGS.checkpoint_path
  pytorch_model = paligemma.build_model(
      checkpoint_path,
      version=int(_VERSION.value),
      custom_loader=loader.maybe_get_custom_loader(
          checkpoint_path, flags.FLAGS.custom_checkpoint_loader
      ),
      mask_cache_size=converter.get_mask_cache_size_from_flags(),
  )

  config = pytorch_model.image_encoder.config.image_embedding
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=f'{flags.FLAGS.output_name_prefix}_{_VERSION.value}',
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
      pixel_values_size=torch.Size(
          [1, config.channels, config.image_size, config.image_size]
      ),
      pixel_seq_len=(config.image_size // config.patch_size) ** 2,
      quantize=flags.FLAGS.quantize,
      config=pytorch_model.config.decoder_config,
      export_config=export_config.get_from_flags(),
  )


if __name__ == '__main__':
  app.run(main)
