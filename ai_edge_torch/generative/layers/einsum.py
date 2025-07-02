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
# Einsum layer implementation.

from typing import Callable, Sequence
import torch
from torch import nn


class Einsum(nn.Module):
  """Einsum layer wrapping over torch.einsum."""

  def __init__(
      self,
      shape: Sequence[int],
      einsum_str: str,
      init_fn: Callable[..., torch.Tensor] = lambda *args, **kwargs: None,
  ):
    super().__init__()
    self.shape = shape
    self.einsum_str = einsum_str
    self.w = nn.Parameter(
        torch.empty(shape, dtype=torch.float32),
        requires_grad=False,
    )
    init_fn(self.w)
    self.einsum_fn = lambda x: torch.einsum(einsum_str, x, self.w)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the Einsum layer."""
    return self.einsum_fn(x)
