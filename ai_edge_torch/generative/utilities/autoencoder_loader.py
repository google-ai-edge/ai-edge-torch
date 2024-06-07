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
# Common utility functions for data loading etc.
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

import ai_edge_torch.generative.layers.model_config as layers_config
import ai_edge_torch.generative.layers.unet.model_config as unet_config
import ai_edge_torch.generative.utilities.loader as loader


@dataclass
class ResidualBlockTensorNames:
  norm_1: str = None
  conv_1: str = None
  norm_2: str = None
  conv_2: str = None
  residual_layer: str = None


@dataclass
class AttnetionBlockTensorNames:
  norm: str = None
  fused_qkv_proj: str = None
  output_proj: str = None


@dataclass
class MidBlockTensorNames:
  residual_block_tensor_names: List[ResidualBlockTensorNames]
  attention_block_tensor_names: List[AttnetionBlockTensorNames]


@dataclass
class UpDecoderBlockTensorNames:
  residual_block_tensor_names: List[ResidualBlockTensorNames]
  upsample_conv: str = None


def _map_to_converted_state(
    state: Dict[str, torch.Tensor],
    state_param: str,
    converted_state: Dict[str, torch.Tensor],
    converted_state_param: str,
):
  converted_state[f"{converted_state_param}.weight"] = state.pop(
      f"{state_param}.weight"
  )
  if f"{state_param}.bias" in state:
    converted_state[f"{converted_state_param}.bias"] = state.pop(f"{state_param}.bias")


class AutoEncoderModelLoader(loader.ModelLoader):

  @dataclass
  class TensorNames:
    quant_conv: str = None
    post_quant_conv: str = None
    conv_in: str = None
    conv_out: str = None
    final_norm: str = None
    mid_block_tensor_names: MidBlockTensorNames = None
    up_decoder_blocks_tensor_names: List[UpDecoderBlockTensorNames] = None

  def __init__(self, file_name: str, names: TensorNames):
    """AutoEncoderModelLoader constructor. Can be used to load encoder and decoder models.

    Args:
        file_name (str): Path to the checkpoint. Can be a directory or an
          exact file.
        names (TensorNames): An instance of `TensorNames` to determine mappings.
    """
    self._file_name = file_name
    self._names = names
    self._loader = self._get_loader()

  def load(
      self, model: torch.nn.Module, strict: bool = True
  ) -> Tuple[List[str], List[str]]:
    """Load the model from the checkpoint.

    Args:
        model (torch.nn.Module): The pytorch model that needs to be loaded.
        strict (bool, optional): Whether the converted keys are strictly
          matched. Defaults to True.

    Returns:
        missing_keys (List[str]): a list of str containing the missing keys.
        unexpected_keys (List[str]): a list of str containing the unexpected keys.

    Raises:
        ValueError: If conversion results in unmapped tensors and strict mode is
          enabled.
    """
    state = self._loader(self._file_name)
    converted_state = dict()
    if self._names.quant_conv is not None:
      _map_to_converted_state(
          state, self._names.quant_conv, converted_state, "quant_conv"
      )
    if self._names.post_quant_conv is not None:
      _map_to_converted_state(
          state, self._names.post_quant_conv, converted_state, "post_quant_conv"
      )
    if self._names.conv_in is not None:
      _map_to_converted_state(state, self._names.conv_in, converted_state, "conv_in")
    if self._names.conv_out is not None:
      _map_to_converted_state(state, self._names.conv_out, converted_state, "conv_out")
    if self._names.final_norm is not None:
      _map_to_converted_state(
          state, self._names.final_norm, converted_state, "final_norm"
      )
    self._map_mid_block(
        state,
        converted_state,
        model.config.mid_block_config,
        self._names.mid_block_tensor_names,
    )

    reversed_block_out_channels = list(reversed(model.config.block_out_channels))
    block_out_channels = reversed_block_out_channels[0]
    for i, out_channels in enumerate(reversed_block_out_channels):
      prev_output_channel = block_out_channels
      block_out_channels = out_channels
      not_final_block = i < len(reversed_block_out_channels) - 1
      self._map_up_decoder_block(
          state,
          converted_state,
          f"up_decoder_blocks.{i}",
          unet_config.UpDecoderBlock2DConfig(
              in_channels=prev_output_channel,
              out_channels=block_out_channels,
              normalization_config=model.config.normalization_config,
              activation_type=model.config.activation_type,
              num_layers=model.config.layers_per_block,
              add_upsample=not_final_block,
              upsample_conv=True,
          ),
          self._names.up_decoder_blocks_tensor_names[i],
      )
    if strict and state:
      raise ValueError(
          f"Failed to map all tensor. Remaing tensor are: {list(state.keys())}"
      )
    return model.load_state_dict(converted_state, strict=strict)

  def _map_residual_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: ResidualBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.ResidualBlock2DConfig,
  ):
    _map_to_converted_state(
        state,
        tensor_names.norm_1,
        converted_state,
        f"{converted_state_param_prefix}.norm_1",
    )
    _map_to_converted_state(
        state,
        tensor_names.conv_1,
        converted_state,
        f"{converted_state_param_prefix}.conv_1",
    )
    _map_to_converted_state(
        state,
        tensor_names.norm_2,
        converted_state,
        f"{converted_state_param_prefix}.norm_2",
    )
    _map_to_converted_state(
        state,
        tensor_names.conv_2,
        converted_state,
        f"{converted_state_param_prefix}.conv_2",
    )
    if config.in_channels != config.out_channels:
      _map_to_converted_state(
          state,
          tensor_names.residual_layer,
          converted_state,
          f"{converted_state_param_prefix}.residual_layer",
      )

  def _map_attention_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      tensor_names: AttnetionBlockTensorNames,
      converted_state_param_prefix: str,
      config: unet_config.AttentionBlock2DConfig,
  ):
    if config.normalization_config.type != layers_config.NormalizationType.NONE:
      _map_to_converted_state(
          state,
          tensor_names.norm,
          converted_state,
          f"{converted_state_param_prefix}.norm",
      )
    attention_layer_prefix = f"{converted_state_param_prefix}.attention"
    _map_to_converted_state(
        state,
        tensor_names.fused_qkv_proj,
        converted_state,
        f"{attention_layer_prefix}.qkv_projection",
    )
    _map_to_converted_state(
        state,
        tensor_names.output_proj,
        converted_state,
        f"{attention_layer_prefix}.output_projection",
    )

  def _map_mid_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      config: unet_config.MidBlock2DConfig,
      tensor_names: MidBlockTensorNames,
  ):
    converted_state_param_prefix = "mid_block"
    residual_block_config = unet_config.ResidualBlock2DConfig(
        in_channels=config.in_channels,
        out_channels=config.in_channels,
        time_embedding_channels=config.time_embedding_channels,
        normalization_config=config.normalization_config,
        activation_type=config.activation_type,
    )
    self._map_residual_block(
        state,
        converted_state,
        tensor_names.residual_block_tensor_names[0],
        f"{converted_state_param_prefix}.resnets.0",
        residual_block_config,
    )
    for i in range(config.num_layers):
      if config.attention_block_config:
        self._map_attention_block(
            state,
            converted_state,
            tensor_names.attention_block_tensor_names[i],
            f"{converted_state_param_prefix}.attentions.{i}",
            config.attention_block_config,
        )
      self._map_residual_block(
          state,
          converted_state,
          tensor_names.residual_block_tensor_names[i + 1],
          f"{converted_state_param_prefix}.resnets.{i+1}",
          residual_block_config,
      )

  def _map_up_decoder_block(
      self,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
      converted_state_param_prefix: str,
      config: unet_config.UpDecoderBlock2DConfig,
      tensor_names: UpDecoderBlockTensorNames,
  ):
    for i in range(config.num_layers):
      input_channels = config.in_channels if i == 0 else config.out_channels
      self._map_residual_block(
          state,
          converted_state,
          tensor_names.residual_block_tensor_names[i],
          f"{converted_state_param_prefix}.resnets.{i}",
          unet_config.ResidualBlock2DConfig(
              in_channels=input_channels,
              out_channels=config.out_channels,
              time_embedding_channels=config.time_embedding_channels,
              normalization_config=config.normalization_config,
              activation_type=config.activation_type,
          ),
      )
    if config.add_upsample and config.upsample_conv:
      _map_to_converted_state(
          state,
          tensor_names.upsample_conv,
          converted_state,
          f"{converted_state_param_prefix}.upsample_conv",
      )
