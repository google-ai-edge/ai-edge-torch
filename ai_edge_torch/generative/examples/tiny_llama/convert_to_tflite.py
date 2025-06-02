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

"""Example of converting TinyLlama model to multi-signature tflite model."""

from absl import app
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
from ai_edge_torch.generative.utilities import loader

flags = converter.define_conversion_flags("tiny_llama")


def main(_):
  checkpoint_path = flags.FLAGS.checkpoint_path
  pytorch_model = tiny_llama.build_model(
      checkpoint_path,
      custom_loader=loader.maybe_get_custom_loader(
          checkpoint_path, flags.FLAGS.custom_checkpoint_loader
      ),
      mask_cache_size=converter.get_mask_cache_size_from_flags(),
  )
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=flags.FLAGS.output_name_prefix,
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
      quantize=flags.FLAGS.quantize,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=export_config.get_from_flags(),
  )


if __name__ == '__main__':
  app.run(main)
