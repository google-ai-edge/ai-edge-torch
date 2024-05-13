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
# Builder class for individual components.
from torch import nn
import torch.nn.functional as F

import ai_edge_torch.generative.layers.feed_forward as feed_forward
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.normalization as normalization


def build_norm(dim: int, config: cfg.NormalizationConfig):
  """Builder function for normalizers.

  Args:
    dim (int): dimension of the input tensor.
    config (`NormalizationConfig` object): the normalization configuration.

  Returns:
    The constructed `nn.Module` normalization layer.

  Raises:
    ValueError: If config's `layer_norm_type` is not supported.
  """
  if config.type == cfg.NormalizationType.NONE:
    return lambda x: x
  elif config.type == cfg.NormalizationType.RMS_NORM:
    return normalization.RMSNorm(
        dim,
        eps=config.epsilon,
        zero_centered_gamma=config.zero_centered,
    )
  elif config.type == cfg.NormalizationType.LAYER_NORM:
    return nn.LayerNorm(dim, eps=config.epsilon)
  else:
    raise ValueError("Unsupported norm type.")


def build_ff(dim: int, config: cfg.FeedForwardConfig):
  """Builder function for Feed Forward. Supports `Sequential` and `Gated`.

  Args:
    dim (int): dimension of the input tensor.
    config (`ModelConfig` object): the model configuration.

  Returns:
    The constructed `nn.Module` feedforward layer.

  Raises:
    ValueError: If config's `ff_type` is not supported.
  """
  ff_type = config.type
  if ff_type == cfg.FeedForwardType.SEQUENTIAL:
    ff_module = feed_forward.SequentialFeedForward
  elif ff_type == cfg.FeedForwardType.GATED:
    ff_module = feed_forward.GatedFeedForward
  else:
    raise ValueError("Unsupported feedforward type.")

  activation = _get_activation(config.activation)

  return ff_module(
      dim=dim,
      hidden_dim=config.intermediate_size,
      activation=activation,
      use_bias=config.use_bias,
  )


def _get_activation(type_: cfg.ActivationType):
  """Get pytorch callable activation from the name.

  Args:
    name (string): activation's name.

  Returns:
    Activation function.

  Raises:
    ValueError: If activation name is not supported.
  """
  if type_ == cfg.ActivationType.SILU:
    return F.silu
  elif type_ == cfg.ActivationType.GELU:
    return F.gelu
  elif type_ == cfg.ActivationType.GELU_TANH:
    return lambda x: F.gelu(x, approximate="tanh")
  elif type_ == cfg.ActivationType.RELU:
    return F.relu
  else:
    raise ValueError("Unsupported activation type.")
