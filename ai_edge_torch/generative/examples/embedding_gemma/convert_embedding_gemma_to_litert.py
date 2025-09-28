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
from ai_edge_torch.generative.examples.embedding_gemma import embedding_gemma
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import loader

flags = converter.define_conversion_flags(
    'embedding_gemma',
    default_mask_as_input=False,
    default_transpose_kv_cache=False,
)

_NORMALIZE_OUTPUT = flags.DEFINE_bool(
    'normalize_output', True, 'Whether to normalize the output with L2 norm.'
)

_SEQ_LEN = flags.DEFINE_integer(
    'seq_len', 2048, 'The sequence length of the model.'
)


def main(_):
  checkpoint_path = flags.FLAGS.checkpoint_path
  pytorch_model = embedding_gemma.build_embedding_gemma(
      checkpoint_path,
      normalize_output=_NORMALIZE_OUTPUT.value,
      custom_loader=loader.maybe_get_custom_loader(
          checkpoint_path, flags.FLAGS.custom_checkpoint_loader
      ),
      mask_cache_size=converter.get_mask_cache_size_from_flags(),
  )
  embedding_gemma.convert_to_litert(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=flags.FLAGS.output_name_prefix,
      prefill_seq_len=_SEQ_LEN.value,
      quantize=flags.FLAGS.quantize,
  )


if __name__ == '__main__':
  app.run(main)
