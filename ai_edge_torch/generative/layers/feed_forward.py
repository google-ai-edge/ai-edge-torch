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
# Common building blocks for FeedForward layers.

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class SequentialFeedForward(nn.Module):
  """Vanilla sequential Feedforward with customizable activation."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      use_bias=False,
  ):
    """Init function for feedforward layer.

    Args:
      dim(int): embedding size.
      hidden_dim(int): hidden dim size of the feedforward layer.
      activation(Callable): activation function used in this block.
      use_bias(Boolean): whether to use bias. Default is false.
    """
    super().__init__()
    self.act = activation
    self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
    self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)

  def forward(self, x):
    """Forward pass for Feedforward layer.

    Args:
      x (torch.Tensor): the input tensor.

    Returns:
      torch.Tensor: output tensor after feedforward.
    """
    return self.w2(self.act(self.w1(x)))


class GatedFeedForward(nn.Module):
  """Gated Feedforward with customizable activation.

  https://arxiv.org/pdf/2002.05202v1.pdf
  """

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      use_bias=False,
  ):
    """Init function for feedforward layer.

    Args:
      dim(int): embedding size.
      hidden_dim(int): hidden dim size of the feedforward layer.
      activation(Callable): activation function used in this block.
      use_bias(Boolean): whether to use bias. Default is false.
    """
    super().__init__()
    self.act = activation
    self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
    self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
    self.w3 = nn.Linear(dim, hidden_dim, bias=use_bias)

  def forward(self, x):
    """Forward pass for Feedforward layer.

    Args:
      x (torch.Tensor): the input tensor.

    Returns:
      torch.Tensor: output tensor after feedforward.
    """
    return self.w2(self.act(self.w1(x)) * self.w3(x))
