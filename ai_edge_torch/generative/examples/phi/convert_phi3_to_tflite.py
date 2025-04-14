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

"""Example of converting a Phi-3.5 model to multi-signature tflite model."""

from absl import app
from ai_edge_torch.generative.examples.phi import phi3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config

flags = converter.define_conversion_flags("phi3")
ExportConfig = export_config.ExportConfig


def main(_):
  pytorch_model = phi3.build_model(
      flags.FLAGS.checkpoint_path, kv_cache_max_len=flags.FLAGS.kv_cache_max_len
  )
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=flags.FLAGS.output_name_prefix,
      prefill_seq_len=flags.FLAGS.prefill_seq_lens,
      quantize=flags.FLAGS.quantize,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=ExportConfig(),
  )


if __name__ == '__main__':
  app.run(main)
