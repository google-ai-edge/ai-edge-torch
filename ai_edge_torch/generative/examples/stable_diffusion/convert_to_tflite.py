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

import os
from pathlib import Path

import torch

import ai_edge_torch
import ai_edge_torch.generative.examples.stable_diffusion.clip as clip
import ai_edge_torch.generative.examples.stable_diffusion.decoder as decoder
import ai_edge_torch.generative.examples.stable_diffusion.diffusion as diffusion
from ai_edge_torch.generative.examples.stable_diffusion.encoder import Encoder
import ai_edge_torch.generative.examples.stable_diffusion.util as util
import ai_edge_torch.generative.utilities.loader as loading_utils
import ai_edge_torch.generative.utilities.stable_diffusion_loader as stable_diffusion_loader


@torch.inference_mode
def convert_stable_diffusion_to_tflite(
    clip_ckpt_path: str,
    encoder_ckpt_path: str,
    diffusion_ckpt_path: str,
    decoder_ckpt_path: str,
    image_height: int = 512,
    image_width: int = 512,
):

  clip_model = clip.CLIP(clip.get_model_config())
  loader = loading_utils.ModelLoader(clip_ckpt_path, clip.TENSOR_NAMES)
  loader.load(clip_model, strict=False)

  encoder = Encoder()
  encoder.load_state_dict(torch.load(encoder_ckpt_path))

  diffusion_model = diffusion.Diffusion(diffusion.get_model_config(2))
  diffusion_loader = stable_diffusion_loader.DiffusionModelLoader(
      diffusion_ckpt_path, diffusion.TENSORS_NAMES
  )
  diffusion_loader.load(diffusion_model)

  decoder_model = decoder.Decoder(decoder.get_model_config())
  decoder_loader = stable_diffusion_loader.AutoEncoderModelLoader(
      decoder_ckpt_path, decoder.TENSORS_NAMES
  )
  decoder_loader.load(decoder_model)

  # Tensors used to trace the model graph during conversion.
  n_tokens = 77
  timestamp = 0
  len_prompt = 1
  prompt_tokens = torch.full((1, n_tokens), 0, dtype=torch.long)
  input_image = torch.full((1, 3, image_height, image_width), 0, dtype=torch.float32)
  noise = torch.full(
      (len_prompt, 4, image_height // 8, image_width // 8), 0, dtype=torch.float32
  )

  input_latents = encoder(input_image, noise)
  context_cond = clip_model(prompt_tokens)
  context_uncond = torch.zeros_like(context_cond)
  context = torch.cat([context_cond, context_uncond], axis=0)
  time_embedding = util.get_time_embedding(timestamp)

  # CLIP text encoder
  ai_edge_torch.signature('encode', clip_model, (prompt_tokens,)).convert().export(
      '/tmp/stable_diffusion/clip.tflite'
  )

  # TODO(yichunk): convert to multi signature tflite model.
  # Image encoder
  ai_edge_torch.signature('encode', encoder, (input_image, noise)).convert().export(
      '/tmp/stable_diffusion/encoder.tflite'
  )

  # Diffusion
  ai_edge_torch.signature(
      'diffusion',
      diffusion_model,
      (torch.repeat_interleave(input_latents, 2, 0), context, time_embedding),
  ).convert().export('/tmp/stable_diffusion/diffusion.tflite')

  # Image decoder
  ai_edge_torch.signature('decode', decoder_model, (input_latents,)).convert().export(
      '/tmp/stable_diffusion/decoder.tflite'
  )


if __name__ == '__main__':
  convert_stable_diffusion_to_tflite(
      clip_ckpt_path=os.path.join(
          Path.home(), 'Downloads/stable_diffusion_data/ckpt/clip.pt'
      ),
      encoder_ckpt_path=os.path.join(
          Path.home(), 'Downloads/stable_diffusion_data/ckpt/encoder.pt'
      ),
      diffusion_ckpt_path=os.path.join(
          Path.home(), 'Downloads/stable_diffusion_data/ckpt/diffusion.pt'
      ),
      decoder_ckpt_path=os.path.join(
          Path.home(), 'Downloads/stable_diffusion_data/ckpt/decoder.pt'
      ),
      image_height=512,
      image_width=512,
  )
