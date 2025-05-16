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

"""Example of building an image encoder of Qwen 2.5 VL model."""

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import attention_utils
from ai_edge_torch.generative.layers import builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn
import torch.nn.functional as F

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="visual.blocks.{}.mlp.up_proj",
    ff_down_proj="visual.blocks.{}.mlp.down_proj",
    ff_gate_proj="visual.blocks.{}.mlp.gate_proj",
    attn_fused_qkv_proj="visual.blocks.{}.attn.qkv",
    attn_output_proj="visual.blocks.{}.attn.proj",
    pre_attn_norm="visual.blocks.{}.norm1",
    post_attn_norm="visual.blocks.{}.norm2",
    embedding="visual.patch_embed.proj",
    final_norm="visual.merger.ln_q",
)

MERGER_TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="visual.merger.mlp.0",
    ff_down_proj="visual.merger.mlp.2",
)


@dataclasses.dataclass
class QwenVLMergerConfig:
  """Merger parameters."""

  activation: cfg.ActivationConfig
  intermediate_size: int
  out_embedding_dim: int
  use_bias: bool = False


@dataclasses.dataclass
class QwenVLImageConfig(cfg.ModelConfig):
  """model config for Qwen 2.5 VL model."""

  merger_config: Optional[QwenVLMergerConfig] = None
  window_size: Optional[int] = None
  spatial_merge_size: Optional[int] = None
  full_atten_block_indexes: Optional[list[int]] = None


class QwenVLMerger(nn.Module):
  """Merger of Qwen 2.5 VL models from the Edge Generative API.

  It's based on Qwen2_5_VLPatchMerger.
  """

  def __init__(self, config: QwenVLImageConfig):
    super().__init__()
    self.intermediate_size = config.merger_config.intermediate_size
    self.w1 = nn.Linear(self.intermediate_size, self.intermediate_size)
    self.act = builder.get_activation(config.merger_config.activation)
    self.w2 = nn.Linear(
        self.intermediate_size, config.merger_config.out_embedding_dim
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_reshaped = x.view(-1, self.intermediate_size)
    return self.w2(self.act(self.w1(x_reshaped)))


class QwenVLImageEncoder(nn.Module):
  """Image encoder of Qwen 2.5 VL models from the Edge Generative API."""

  def __init__(self, config: QwenVLImageConfig):
    super().__init__()

    # Tensor shape used to reshape pixel_values in forward() and various places.
    self.kernel_size = (
        -1,  # pixel_values.size(0)
        config.image_embedding.channels,
        config.image_embedding.temporal_patch_size,
        config.image_embedding.patch_size,
        config.image_embedding.patch_size,
    )
    self.tok_embedding = nn.Conv3d(
        in_channels=self.kernel_size[1],
        out_channels=config.embedding_dim,
        kernel_size=self.kernel_size[2:],
        stride=self.kernel_size[2:],
        padding=0,
        bias=config.embedding_use_bias,
    )

    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.merger = QwenVLMerger(config)
    self.config = config
    self.set_image_size(config.image_embedding.image_size)

  @torch.inference_mode
  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    # Check if the pixel value size matches with grid size and image config.
    assert pixel_values.size() == self.get_pixel_values_size(self.grid_thw)

    # Embed the image and rearrange the embedding tensor.
    pixel_reshaped = pixel_values.reshape(self.kernel_size)
    x = self.tok_embedding(pixel_reshaped)
    x = x.view(-1, self.config.embedding_dim)
    x = self._rearrange(x, self.window_index).unsqueeze(0)

    rope = self._get_rope(self.grid_thw, self.window_index)

    mask = self._get_mask(self.grid_thw, self.cu_seqlens)
    full_mask = torch.zeros(x.shape[:2])
    for i, block in enumerate(self.transformer_blocks):
      x = block(
          x,
          rope=rope,
          mask=full_mask if i in self.config.full_atten_block_indexes else mask,
      )

    y = self.merger.forward(self.final_norm(x))
    # Arrange the output back to the original order.
    return y[self.reverse_index, ...]

  def set_image_size(self, image_size: Tuple[int, int]):
    """Set the image size and pre-calculate some values including mask."""
    self.config.image_embedding.image_size = image_size
    self.grid_thw = self.get_grid_thw()

    # Precalculate the window index which can't be lowered to HLO because of
    # inconcrete index in:
    #     index_new = index_padded[index_padded != -100]
    self.window_index, self.cu_seqlens = self._get_window_index(self.grid_thw)

    # Precalculate the reverse index of window_index until "vhlo.sort_v1" op is
    # supported.
    self.reverse_index = torch.argsort(self.window_index)

  def get_grid_thw(self, num_images: int = 1) -> List[Tuple[int, int, int]]:
    """Calculate the grid size of the input images based on the image config."""
    height, width = self.config.image_embedding.image_size
    patch_height = height // self.config.image_embedding.patch_size
    patch_width = width // self.config.image_embedding.patch_size
    # Support only image, i.e. temporal step size is always 1.
    return [(1, patch_height, patch_width)] * num_images

  def get_pixel_values_size(
      self, grid_thw: List[Tuple[int, int, int]]
  ) -> torch.Size:
    """Calculate the size of pixel values tensor."""
    dim_0 = sum(t * h * w for t, h, w in grid_thw)
    config = self.config.image_embedding
    dim_1 = config.channels * config.temporal_patch_size * config.patch_size**2
    return torch.Size((dim_0, dim_1))

  def _get_rope(
      self, grid_thw: List[Tuple[int, int, int]], window_index: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get RoPE for Qwen VL model based on image grid information.

    It's copied from Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb() and
    modified accordingly.
    """
    pos_ids = []
    for t, h, w in grid_thw:
      hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
      hpos_ids = hpos_ids.reshape(
          h // self.config.spatial_merge_size,
          self.config.spatial_merge_size,
          w // self.config.spatial_merge_size,
          self.config.spatial_merge_size,
      )
      hpos_ids = hpos_ids.permute(0, 2, 1, 3)
      hpos_ids = hpos_ids.flatten()

      wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
      wpos_ids = wpos_ids.reshape(
          h // self.config.spatial_merge_size,
          self.config.spatial_merge_size,
          w // self.config.spatial_merge_size,
          self.config.spatial_merge_size,
      )
      wpos_ids = wpos_ids.permute(0, 2, 1, 3)
      wpos_ids = wpos_ids.flatten()
      pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    # Assume all the heights and widths are the same for all images.
    max_grid_size = max(grid_thw[0][1], grid_thw[0][2])

    cos, sin = attention_utils.build_rope_cache(
        max_grid_size,
        # ROPE parameters for all attn_configs are the same. Take the first one.
        self.config.block_config(0).attn_config.head_dim // 2,
    )
    return (
        self._rearrange(cos[pos_ids].flatten(1), window_index),
        self._rearrange(sin[pos_ids].flatten(1), window_index),
    )

  def _get_window_index(self, grid_thw: List[Tuple[int, int, int]]):
    """Get window index for Qwen VL model to rearrange the input tensor.

    It's copied from Qwen2_5_VisionTransformerPretrainedModel.get_window_index()
    and modified accordingly.
    """
    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = (
        self.config.window_size
        // self.config.spatial_merge_size
        // self.config.image_embedding.patch_size
    )

    for grid_t, grid_h, grid_w in grid_thw:
      llm_grid_h = grid_h // self.config.spatial_merge_size
      llm_grid_w = grid_w // self.config.spatial_merge_size
      index = torch.arange(grid_t * llm_grid_h * llm_grid_w)
      index = index.reshape((grid_t, llm_grid_h, llm_grid_w))
      pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
      pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
      num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
      num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
      index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
      index_padded = index_padded.reshape(
          grid_t,
          num_windows_h,
          vit_merger_window_size,
          num_windows_w,
          vit_merger_window_size,
      )
      index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
          grid_t,
          num_windows_h * num_windows_w,
          vit_merger_window_size,
          vit_merger_window_size,
      )
      seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
      index_padded = index_padded.reshape(-1)
      index_new = index_padded[index_padded != -100]
      window_index.append(index_new + window_index_id)
      spatial_merge_unit = self.config.spatial_merge_size**2
      cu_seqlens_tmp = (
          seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
      )
      cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
      window_index_id += grid_t * llm_grid_h * llm_grid_w

    window_index = torch.cat(window_index, dim=0)
    return window_index, cu_window_seqlens

  def _rearrange(
      self, x: torch.Tensor, window_index: torch.Tensor
  ) -> torch.Tensor:
    """Rearrange the tensor according to window_index.

    It's copied from Qwen2_5_VisionTransformerPretrainedModel.forward() and
    modified accordingly.
    """
    spatial_merge_unit = self.config.spatial_merge_size**2
    x_reshaped = x.view(x.size(0) // spatial_merge_unit, spatial_merge_unit, -1)
    x_rearranged = x_reshaped[window_index, ...]
    return x_rearranged.view(x.shape)

  def _get_mask(
      self, grid_thw: List[Tuple[int, int, int]], cu_seqlens: List[int]
  ) -> torch.Tensor:
    """Get attention mask for Qwen VL model.

    It's copied from Qwen2_5_VLVisionAttention.forward() and modified
    accordingly.
    """
    seqlen = self.get_pixel_values_size(grid_thw)[0]
    mask = torch.full([1, 1, seqlen, seqlen], float("-inf"))
    for i in range(1, len(cu_seqlens)):
      mask[
          ...,
          cu_seqlens[i - 1] : cu_seqlens[i],
          cu_seqlens[i - 1] : cu_seqlens[i],
      ] = 0
    return mask


def get_image_encoder_config(image_size: Tuple[int, int]) -> QwenVLImageConfig:
  """Returns the model config for the image encoder of a Qwen 2.5 VL model.

  Returns:
    The model config for the image encoder of a Qwen 2.5 VL model.
  """
  image_embedding_config = cfg.ImageEmbeddingConfig(
      channels=3,
      image_size=image_size,
      patch_size=14,
      temporal_patch_size=2,
  )
  attn_config = cfg.AttentionConfig(
      num_heads=16,
      head_dim=80,
      num_query_groups=16,
      qkv_transpose_before_split=True,
      qkv_use_bias=True,
      output_proj_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=3420,
      use_bias=True,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  merger_config = QwenVLMergerConfig(
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU),
      intermediate_size=5120,  # embedding_dim(1280) * spatial_merge_size(2)^2
      out_embedding_dim=2048,  # embedding_dim of decoder config.
      use_bias=True,
  )
  config = QwenVLImageConfig(
      vocab_size=0,  # Not used in image encoder.
      num_layers=32,
      max_seq_len=0,  # Not used in image encoder.
      embedding_dim=1280,
      image_embedding=image_embedding_config,
      block_configs=block_config,
      final_norm_config=norm_config,
      merger_config=merger_config,
      window_size=112,
      spatial_merge_size=2,
      full_atten_block_indexes=[7, 15, 23, 31],
  )
  return config


def get_fake_image_encoder_config() -> QwenVLImageConfig:
  config = get_image_encoder_config((8, 12))
  # PaliGemma image encoder has only one block config.
  config.block_config(0).ff_config.intermediate_size = 128
  config.image_embedding.patch_size = 2
  config.num_layers = 2
  config.merger_config.intermediate_size = 128
  return config


def build_image_encoder(
    checkpoint_path: str,
    image_size: Tuple[int, int] = (34 * 14, 46 * 14),
) -> QwenVLImageEncoder:
  config = get_image_encoder_config(image_size)
  encoder = QwenVLImageEncoder(config)
  load_image_encoder(checkpoint_path, encoder)
  encoder.eval()
  return encoder


def load_image_encoder(
    checkpoint_path: str,
    encoder: QwenVLImageEncoder,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
):
  loader = loading_utils.ModelLoader(
      checkpoint_path, TENSOR_NAMES, custom_loader
  )
  # Loose the strictness because only image encoder is being loaded.
  loader.load(encoder, strict=False)

  # Load merger weights.
  merger_loader = loading_utils.ModelLoader(
      checkpoint_path, None, custom_loader
  )
  state = merger_loader.get_state()
  w1_state = dict()
  w1_state["weight"] = state.pop(f"{MERGER_TENSOR_NAMES.ff_up_proj}.weight")
  if encoder.config.merger_config.use_bias:
    w1_state["bias"] = state.pop(f"{MERGER_TENSOR_NAMES.ff_up_proj}.bias")
  encoder.merger.w1.load_state_dict(w1_state)

  w2_state = dict()
  w2_state["weight"] = state.pop(f"{MERGER_TENSOR_NAMES.ff_down_proj}.weight")
  if encoder.config.merger_config.use_bias:
    w2_state["bias"] = state.pop(f"{MERGER_TENSOR_NAMES.ff_down_proj}.bias")
  encoder.merger.w2.load_state_dict(w2_state)
