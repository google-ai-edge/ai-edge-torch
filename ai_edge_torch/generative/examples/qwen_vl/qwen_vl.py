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

"""Example of building a full-stack of Qwen 2.5 VL model."""

import dataclasses
from typing import List, Optional, Tuple

from ai_edge_torch.generative.examples.qwen_vl import decoder
from ai_edge_torch.generative.examples.qwen_vl import image_encoder
import ai_edge_torch.generative.layers.kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


@dataclasses.dataclass
class QwenVLConfig:
  """Qwen VL model configurations."""

  image_encoder_config: image_encoder.QwenVLImageConfig
  decoder_config: cfg.ModelConfig
  image_token_id: int
  mrope_section: List[int]


class QwenVL(nn.Module):
  """Qwen VL model from the Edge Generative API."""

  def __init__(self, config: QwenVLConfig):
    super().__init__()

    self.image_encoder = image_encoder.QwenVLImageEncoder(
        config.image_encoder_config
    )
    self.decoder = decoder.Decoder(config.decoder_config)
    # The amount of adjustment in input_pos to calculate RoPE properly in
    # forward() calls after image is handled.
    self.rope_pos_adjust = 0
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      pixel_values: torch.Tensor = None,
      export_config: Optional[model_builder.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if pixel_values is None:
      return self.decoder(
          tokens=tokens,
          input_pos=input_pos,
          kv_cache=kv_cache,
          input_embeds=None,
          rope=self._build_text_rope(input_pos),
          mask=mask,
          export_config=export_config,
      )

    input_embeds = self.decoder.tok_embedding(tokens)
    image_embeds = self.image_encoder(pixel_values).unsqueeze(0)

    # Merging image_embeds into text_embeds as PaliGemmaForConditionalGeneration
    # can be done like:
    #
    #   image_mask = tokens == self.config.image_token_id
    #   image_mask = image_mask.unsqueeze(-1).expand_as(input_embeds)
    #   input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)
    #
    # Unfortunately, torch.Tensor.masked_scatter can't be lowered on CPU.
    # Assume that image is put at the beginning of the input sequence wrapped
    # with vision_start and vision_end tokens.
    input_embeds = torch.cat(
        (
            input_embeds[:, :1, :],
            image_embeds,
            input_embeds[:, image_embeds.size(1) + 1 :, :],
        ),
        dim=1,
    )

    grid_thw = self.image_encoder.get_grid_thw()
    return self.decoder(
        tokens=None,
        input_pos=input_pos,
        kv_cache=kv_cache,
        input_embeds=input_embeds,
        rope=self._build_multimodal_rope(input_pos, grid_thw),
        mask=mask,
        export_config=export_config,
    )

  def _build_rope(
      self, rope_pos: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # ROPE parameters for all attn_configs are the same. Take the first one.
    attn_config = self.config.decoder_config.block_config(0).attn_config
    n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
    return self.config.decoder_config.build_rope(
        rope_pos, n_elem, attn_config.rotary_base
    )

  def _build_text_rope(
      self, input_pos: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reset rope_pos_adjust to 0 when it's prefill, i.e. input has 2 or more
    # tokens.
    if input_pos.numel() > 1:
      self.rope_pos_adjust = 0
    return self._build_rope(input_pos + self.rope_pos_adjust)

  def _build_multimodal_rope(
      self, input_pos: torch.Tensor, grid_thw: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds RoPE of multimodal input for the Qwen VL model.

    It's copied from Qwen2_5_VLForConditionalGeneration.get_rope_index() and
    simplified based on the assumption that an image is put at the beginning of
    the input sequence with vision start and vision end tokens.
    """
    spatial_merge_size = self.config.image_encoder_config.spatial_merge_size
    height = grid_thw[0][1] // spatial_merge_size
    width = grid_thw[0][2] // spatial_merge_size
    image_pos_max = max(height, width)
    image_pos_count = height * width

    # The position of vision end tokek and text tokens and after the image.
    text_pos_start = image_pos_max + 1
    text_pos_count = len(input_pos) - image_pos_count - 1
    text_pos = torch.arange(text_pos_start, text_pos_start + text_pos_count)
    # Set input_pos_adjust since text_pos_start has changed.
    self.rope_pos_adjust = image_pos_max - image_pos_count

    temporal_rope = self._build_image_text_rope(
        torch.ones(image_pos_count, dtype=torch.int), text_pos
    )
    height_rope = self._build_image_text_rope(
        torch.arange(1, height + 1).view(-1, 1).expand(-1, width).flatten(),
        text_pos,
    )
    width_rope = self._build_image_text_rope(
        torch.arange(1, width + 1).view(1, -1).expand(height, -1).flatten(),
        text_pos,
    )

    return (
        self._merge_ropes(temporal_rope[0], height_rope[0], width_rope[0]),
        self._merge_ropes(temporal_rope[1], height_rope[1], width_rope[1]),
    )

  def _build_image_text_rope(
      self, image_pos: torch.Tensor, text_pos: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    return self._build_rope(
        torch.cat((torch.zeros(1, dtype=torch.int), image_pos, text_pos))
    )

  def _merge_ropes(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """Merges RoPE tensors based on apply_multimodal_rotary_pos_emb()."""
    split = torch.stack([a, b, c]).split(self.config.mrope_section, dim=-1)
    return torch.cat([m[i % 3] for i, m in enumerate(split)], dim=-1)


def get_model_config(
    kv_cache_max_len: int = 1024,
    image_size: Tuple[int, int] = (34 * 14, 46 * 14),
) -> QwenVLConfig:
  """Returns the model config for a PaliGemma 3B-224 model.

  Returns:
    The model config for a PaliGemma 3B model.
  """
  return QwenVLConfig(
      image_encoder_config=image_encoder.get_image_encoder_config(image_size),
      decoder_config=decoder.get_decoder_config(kv_cache_max_len),
      image_token_id=151655,
      mrope_section=[16, 24, 24],
  )


def get_fake_model_config(**kwargs) -> QwenVLConfig:
  return QwenVLConfig(
      image_encoder_config=image_encoder.get_fake_image_encoder_config(),
      decoder_config=decoder.get_fake_decoder_config(**kwargs),
      image_token_id=127,
      mrope_section=[16, 24, 24],
  )


def build_model(checkpoint_path: str, **kwargs) -> QwenVL:
  config = get_model_config(**kwargs)
  model = QwenVL(config)
  image_encoder.load_image_encoder(checkpoint_path, model.image_encoder)
  # Load the parameters of decoder.
  loader = loading_utils.ModelLoader(checkpoint_path, decoder.TENSOR_NAMES)
  loader.load(model.decoder, strict=False)
  model.eval()
  return model
