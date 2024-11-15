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

"""Example of building a full-stack of PaliGemma model."""

from dataclasses import dataclass

from ai_edge_torch.generative.examples.paligemma import decoder
from ai_edge_torch.generative.examples.paligemma import image_encoder
import ai_edge_torch.generative.layers.kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

PROJECTION_TENSOR_NAME = "multi_modal_projector.linear"


@dataclass
class PaliGemmaConfig:
  """PaliGemma model configurations."""

  image_encoder_config: cfg.ModelConfig
  decoder_config: cfg.ModelConfig

  image_token_id: int
  image_projection_use_bias: bool = False


class PaliGemma(nn.Module):
  """PaliGemma model from the Edge Generative API."""

  def __init__(self, config: PaliGemmaConfig):
    super().__init__()

    self.image_encoder = image_encoder.SiglipVisionEncoder(
        config.image_encoder_config
    )
    self.image_projection = nn.Linear(
        config.image_encoder_config.embedding_dim,
        config.decoder_config.embedding_dim,
        bias=config.image_projection_use_bias,
    )
    self.decoder = decoder.Decoder(config.decoder_config)
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      pixel_values: torch.Tensor = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if pixel_values is None:
      return self.decoder(tokens, input_pos, kv_cache)

    input_embeds = self.decoder.tok_embedding(tokens)

    image_encoded = self.image_encoder(pixel_values=pixel_values)
    image_embeds = self.image_projection(image_encoded)
    if self.config.decoder_config.embedding_scale is not None:
      image_embeds = image_embeds / self.config.decoder_config.embedding_scale

    # Merge image_embeds into text_embeds as PaliGemmaForConditionalGeneration.
    image_mask = tokens == self.config.image_token_id
    image_mask = image_mask.unsqueeze(-1).expand_as(input_embeds)
    input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

    return self.decoder(
        tokens=None,
        input_pos=input_pos,
        kv_cache=kv_cache,
        input_embeds=input_embeds,
    )


def get_model_config() -> PaliGemmaConfig:
  """Returns the model config for a PaliGemma 3B-224 model.

  Returns:
    The model config for a PaliGemma 3B model.
  """
  return PaliGemmaConfig(
      image_encoder_config=image_encoder.get_image_encoder_config(),
      decoder_config=decoder.get_decoder_config(),
      image_projection_use_bias=True,
      image_token_id=257152,
  )


def get_fake_image_encoder_config() -> PaliGemmaConfig:
  return PaliGemmaConfig(
      image_encoder_config=image_encoder.get_fake_image_encoder_config(),
      decoder_config=decoder.get_fake_decoder_config(),
      image_projection_use_bias=True,
      image_token_id=257152,
  )


def build_model(checkpoint_path: str) -> PaliGemma:
  config = get_model_config()
  model = PaliGemma(config)
  # Load the parameters of image encoder.
  loader = loading_utils.ModelLoader(
      checkpoint_path, image_encoder.TENSOR_NAMES
  )
  loader.load(model.image_encoder, strict=False)
  # Load the parameters of decoder.
  loader = loading_utils.ModelLoader(checkpoint_path, decoder.TENSOR_NAMES)
  loader.load(model.decoder, strict=False)

  # Load the parameters of image projection.
  loader = loading_utils.ModelLoader(checkpoint_path, None)
  state = loader.get_state()
  converted_state = dict()
  converted_state["weight"] = state.pop(f"{PROJECTION_TENSOR_NAME}.weight")
  if config.image_projection_use_bias:
    converted_state["bias"] = state.pop(f"{PROJECTION_TENSOR_NAME}.bias")
  model.image_projection.load_state_dict(converted_state)

  model.eval()
  return model
