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
# Common normalization layers.

import torch


# Implementation for RMSNorm from: https://arxiv.org/abs/1910.07467
class RMSNorm(torch.nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6, zero_centered_gamma=False):
    """
    Initialize the RMSNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default: 1e-6).
    """
    super().__init__()
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.zero_centered_gamma = zero_centered_gamma

  def _norm(self, x):
    """
    Apply RMSNorm normalization.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: The normalized output tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    """
    Running the forward pass of RMSNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying RMSNorm.
    """
    output = self._norm(x.float()).type_as(x)
    if self.zero_centered_gamma:
      return output * (1 + self.weight)
    else:
      return output * self.weight
