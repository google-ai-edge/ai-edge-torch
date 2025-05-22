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

import abc
from typing import Callable

import ai_edge_torch.generative.layers.model_config as cfg
import torch
from torch import nn


class FeedForwardBase(nn.Module):
  """Base class for feedforward layer."""

  def __init__(
      self,
      dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      config: cfg.FeedForwardConfig,
      pre_ff_norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
      post_ff_norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
  ):
    super().__init__()
    self.dim = dim
    self.act = activation
    self.config = config
    self.hidden_dim = config.intermediate_size
    self.use_bias = config.use_bias
    self.use_glu = (
        config.activation.type == cfg.ActivationType.GE_GLU
        or config.activation.type == cfg.ActivationType.SILU_GLU
    )
    self.pre_ff_norm = pre_ff_norm
    self.post_ff_norm = post_ff_norm

  @abc.abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError()


class SequentialFeedForward(FeedForwardBase):
  """Vanilla sequential Feedforward with customizable activation."""

  def __init__(
      self,
      dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      config: cfg.FeedForwardConfig,
      pre_ff_norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
      post_ff_norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
  ):
    """Init function for feedforward layer.

    Args:
      dim (int): embedding size.
      activation (Callable): activation function used in this block.
      config (cfg.FeedForwardConfig): feedforward layer configuration.
      pre_ff_norm (Callable): pre feedforward norm. Default is identity.
      post_ff_norm (Callable): post feedforward norm. Default is identity.
    """
    super().__init__(dim, activation, config, pre_ff_norm, post_ff_norm)
    if self.use_glu:
      self.w1 = nn.Linear(dim, self.hidden_dim * 2, bias=self.use_bias)
    else:
      self.w1 = nn.Linear(dim, self.hidden_dim, bias=self.use_bias)
    self.w2 = nn.Linear(self.hidden_dim, dim, bias=self.use_bias)

  def forward(self, x):
    """Forward pass for Feedforward layer.

    Args:
      x (torch.Tensor): the input tensor.

    Returns:
      torch.Tensor: output tensor after feedforward.
    """
    x_norm = self.pre_ff_norm(x)
    out = self.w2(self.act(self.w1(x_norm)))
    return self.post_ff_norm(out)


class GatedFeedForward(FeedForwardBase):
  """Gated Feedforward with customizable activation.

  https://arxiv.org/pdf/2002.05202v1.pdf
  """

  def __init__(
      self,
      dim: int,
      activation: Callable[[torch.Tensor], torch.Tensor],
      config: cfg.FeedForwardConfig,
      pre_ff_norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
      post_ff_norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
  ):
    """Init function for feedforward layer.

    Args:
      dim (int): embedding size.
      activation (Callable): activation function used in this block.
      pre_ff_norm (Callable): pre feedforward norm. Default is identity.
      post_ff_norm (Callable): post feedforward norm. Default is identity.
      config (cfg.FeedForwardConfig): feedforward layer configuration.
    """
    super().__init__(dim, activation, config, pre_ff_norm, post_ff_norm)

    if self.use_glu:
      assert (
          self.config.use_separate_gating
      ), 'use_separate_gating must be True for GE_GLU | SILU_GLU activation.'

    if self.config.use_separate_gating:
      if self.use_glu:
        self.w1 = nn.Linear(dim, self.hidden_dim * 2, bias=self.use_bias)
      else:
        self.w1 = nn.Linear(dim, self.hidden_dim, bias=self.use_bias)
      self.w3 = nn.Linear(dim, self.hidden_dim, bias=self.use_bias)
    else:
      self.w_gating = nn.Parameter(
          torch.ones((2, dim, self.hidden_dim), dtype=torch.float32),
          requires_grad=False,
      )
      self.gating_bias = (
          nn.Parameter(
              torch.zeros((2, self.hidden_dim), dtype=torch.float32),
              requires_grad=False,
          )
          if self.use_bias
          else torch.zeros((2, self.hidden_dim), dtype=torch.float32)
      )

    self.w2 = nn.Linear(self.hidden_dim, dim, bias=self.use_bias)

  def forward(self, x):
    """Forward pass for Feedforward layer.

    Args:
      x (torch.Tensor): the input tensor.

    Returns:
      torch.Tensor: output tensor after feedforward.
    """
    x_norm = self.pre_ff_norm(x)
    if self.config.use_separate_gating:
      out = self.w2(self.act(self.w1(x_norm)) * self.w3(x_norm))
    else:
      out = self.w2(
          self.act(torch.matmul(x_norm, self.w_gating[0]) + self.gating_bias[0])
          * (torch.matmul(x_norm, self.w_gating[1]) + self.gating_bias[1])
      )

    return self.post_ff_norm(out)
