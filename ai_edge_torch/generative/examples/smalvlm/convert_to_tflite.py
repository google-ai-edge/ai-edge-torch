from absl import app
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import export_config
import torch

import sys
sys.path.append('/data/usr/dmitry.korostelev/ml-vlms/')

from mobile.smalvlm import smalvlm

flags = converter.define_conversion_flags('smalvlm-256m-instruct')


def main(_):

  prefill_seq_lens = 256
  kv_cache_max_len = 2048
  checkpoint_path="/data/usr/dmitry.korostelev/models/SmolVLM-256M-Instruct"

  pytorch_model = smalvlm.build_model(
    checkpoint_path=checkpoint_path,
    custom_loader=None,
    kv_cache_max_len=kv_cache_max_len,
  )

  config = pytorch_model.image_encoder.config.image_embedding
  converter.convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=f'{flags.FLAGS.output_name_prefix}',
      prefill_seq_len=prefill_seq_lens,
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
