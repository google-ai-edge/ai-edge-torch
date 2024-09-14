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

import argparse
import os
from pathlib import Path
from typing import Optional

import ai_edge_torch
import ai_edge_torch.generative.examples.stable_diffusion.clip as clip
import ai_edge_torch.generative.examples.stable_diffusion.decoder as decoder
import ai_edge_torch.generative.examples.stable_diffusion.diffusion as diffusion
from ai_edge_torch.generative.examples.stable_diffusion.encoder import Encoder
import ai_edge_torch.generative.examples.stable_diffusion.util as util
from ai_edge_torch.generative.quantize import quant_recipes
import ai_edge_torch.generative.utilities.stable_diffusion_loader as stable_diffusion_loader
import torch

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--clip_ckpt',
    type=str,
    help='Path to source CLIP model checkpoint',
    required=True,
)
arg_parser.add_argument(
    '--diffusion_ckpt',
    type=str,
    help='Path to source diffusion model checkpoint',
    required=True,
)
arg_parser.add_argument(
    '--decoder_ckpt',
    type=str,
    help='Path to source image decoder model checkpoint',
    required=True,
)
arg_parser.add_argument(
    '--output_dir',
    type=str,
    help='Path to the converted TF Lite directory.',
    required=True,
)


@torch.inference_mode
def convert_stable_diffusion_to_tflite(
    output_dir: str,
    clip_ckpt_path: str,
    diffusion_ckpt_path: str,
    decoder_ckpt_path: str,
    image_height: int = 512,
    image_width: int = 512,
    quantize: bool = True,
):

  clip_model = clip.CLIP(clip.get_model_config())
  loader = stable_diffusion_loader.ClipModelLoader(
      clip_ckpt_path,
      clip.TENSOR_NAMES,
  )
  loader.load(clip_model, strict=False)

  diffusion_model = diffusion.Diffusion(diffusion.get_model_config(2))
  diffusion_loader = stable_diffusion_loader.DiffusionModelLoader(
      diffusion_ckpt_path, diffusion.TENSOR_NAMES
  )
  diffusion_loader.load(diffusion_model, strict=False)

  decoder_model = decoder.Decoder(decoder.get_model_config())
  decoder_loader = stable_diffusion_loader.AutoEncoderModelLoader(
      decoder_ckpt_path, decoder.TENSOR_NAMES
  )
  decoder_loader.load(decoder_model, strict=False)

  # TODO(yichunk): enable image encoder conversion
  # if encoder_ckpt_path is not None:
  #   encoder = Encoder()
  #   encoder.load_state_dict(torch.load(encoder_ckpt_path))

  # Tensors used to trace the model graph during conversion.
  n_tokens = 77
  timestamp = 0
  len_prompt = 1
  prompt_tokens = torch.full((1, n_tokens), 0, dtype=torch.int)
  input_image = torch.full(
      (1, 3, image_height, image_width), 0, dtype=torch.float32
  )
  noise = torch.full(
      (len_prompt, 4, image_height // 8, image_width // 8),
      0,
      dtype=torch.float32,
  )

  input_latents = torch.zeros_like(noise)
  context_cond = clip_model(prompt_tokens)
  context_uncond = torch.zeros_like(context_cond)
  context = torch.cat([context_cond, context_uncond], axis=0)
  time_embedding = util.get_time_embedding(timestamp)

  if not os.path.exists(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

  quant_config = (
      quant_recipes.full_int8_weight_only_recipe() if quantize else None
  )

  # TODO(yichunk): convert to multi signature tflite model.
  # CLIP text encoder
  ai_edge_torch.signature('encode', clip_model, (prompt_tokens,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/clip.tflite')

  # TODO(yichunk): enable image encoder conversion
  # Image encoder
  # ai_edge_torch.signature('encode', encoder, (input_image, noise)).convert(quant_config=quant_config).export(
  #     f'{output_dir}/encoder.tflite'
  # )

  # Diffusion
  ai_edge_torch.signature(
      'diffusion',
      diffusion_model,
      (torch.repeat_interleave(input_latents, 2, 0), context, time_embedding),
  ).convert(quant_config=quant_config).export(f'{output_dir}/diffusion.tflite')

  # Image decoder
  ai_edge_torch.signature('decode', decoder_model, (input_latents,)).convert(
      quant_config=quant_config
  ).export(f'{output_dir}/decoder.tflite')


if __name__ == '__main__':
  args = arg_parser.parse_args()
  convert_stable_diffusion_to_tflite(
      output_dir=args.output_dir,
      clip_ckpt_path=args.clip_ckpt,
      diffusion_ckpt_path=args.diffusion_ckpt,
      decoder_ckpt_path=args.decoder_ckpt,
      image_height=512,
      image_width=512,
      quantize=True,
  )
