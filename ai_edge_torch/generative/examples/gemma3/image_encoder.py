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

"""Example of building an image encoder of Gemma3 model which is Siglip."""

from ai_edge_torch.generative.examples.paligemma import image_encoder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn
import torch.nn.functional as F


TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="siglip_vision_model.encoder_blocks.{}.mlp.fc1",
    ff_down_proj="siglip_vision_model.encoder_blocks.{}.mlp.fc2",
    attn_query_proj=(
        "siglip_vision_model.encoder_blocks.{}.self_attn.q_proj"
    ),
    attn_key_proj=(
        "siglip_vision_model.encoder_blocks.{}.self_attn.k_proj"
    ),
    attn_value_proj=(
        "siglip_vision_model.encoder_blocks.{}.self_attn.v_proj"
    ),
    attn_output_proj=(
        "siglip_vision_model.encoder_blocks.{}.self_attn.o_proj"
    ),
    pre_attn_norm="siglip_vision_model.encoder_blocks.{}.layer_norm1",
    pre_ff_norm="siglip_vision_model.encoder_blocks.{}.layer_norm2",
    embedding="siglip_vision_model.patch_embedding",
    embedding_position=(
        "siglip_vision_model.position_embedding.weight"
    ),
    final_norm="siglip_vision_model.final_norm",
)


class SiglipExit(nn.Module):
  """Siglip exit layer."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()
    self.expected_length = config.num_mm_tokens_per_image**0.5

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    current_tokens = x.shape[1]
    current_length = int(current_tokens**0.5)
    if current_length != self.expected_length:
      window_size = int(current_length // self.expected_length)
      x = x.transpose(1, 2)
      x = x.view(x.shape[0], x.shape[1], current_length, current_length)
      x = F.avg_pool2d(x, window_size, stride=window_size)
      x = x.view(x.shape[0], x.shape[1], -1)
      x = x.transpose(1, 2)
    return x

class SiglipVisionEncoderWithExit(nn.Module):
  """Siglip vision encoder for Gemma3MM from the Edge Generative API."""

  def __init__(self, config: cfg.ModelConfig):
      super().__init__()
      self.siglip_encoder = image_encoder.SiglipVisionEncoder(config)
      self.siglip_exit = SiglipExit(config)

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    x = self.siglip_encoder(pixel_values)
    x = self.siglip_exit(x)
    return x

def get_image_encoder_config() -> cfg.ModelConfig:
  """Returns the model config for the image encoder of a Gemma3 4B model.

  Returns:
    The model config for the image encoder of a Gemma3 4B model.
  """
  image_embedding_config = cfg.ImageEmbeddingConfig(
      channels=3,
      image_size=896,
      patch_size=14,
  )
  attn_config = cfg.AttentionConfig(
      num_heads=16,
      head_dim=72,
      num_query_groups=16,
      qkv_use_bias=True,
      output_proj_use_bias=True,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.LAYER_NORM, epsilon=1e-6
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=4304,
      use_bias=True,
      pre_ff_norm_config=norm_config,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=0,  # Not used in image encoder.
      num_layers=27,
      max_seq_len=0,  # Not used in image encoder.
      embedding_dim=1152,
      embedding_use_bias=True,
      image_embedding=image_embedding_config,
      block_configs=block_config,
      final_norm_config=norm_config,
      num_mm_tokens_per_image=256,
  )
  return config


def get_fake_image_encoder_config() -> cfg.ModelConfig:
  config = get_image_encoder_config()
  config.block_config(0).ff_config.intermediate_size = 128
  config.image_embedding.image_size = 8
  config.image_embedding.patch_size = 2
  config.num_layers = 2
  config.num_mm_tokens_per_image = 4
  return config


def build_image_encoder(checkpoint_path: str) -> SiglipVisionEncoderWithExit:
  config = get_image_encoder_config()
  encoder = SiglipVisionEncoderWithExit(config).siglip_encoder
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  # Loose the strictness because only image encoder is being loaded.
  loader.load(encoder, strict=False)
  encoder.eval()
  return encoder
