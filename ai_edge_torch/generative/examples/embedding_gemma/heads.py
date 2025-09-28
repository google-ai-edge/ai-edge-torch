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

"""Embedding Gemma Heads layers."""

import torch


def _divide_no_nan(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
  y_is_zero = torch.abs(y) <= max(eps, 7e-5)
  safe_y = torch.where(y_is_zero, 1, y)
  return torch.where(y_is_zero, torch.zeros_like(x), torch.divide(x, safe_y))


class MeanPooling(torch.nn.Module):
  """Mean pooling."""

  def forward(
      self,
      x_unpooled: torch.Tensor,  # [B, T, D]
      input_mask: torch.Tensor,  # [*B, T]
  ) -> torch.Tensor:
    input_mask_expand = input_mask.unsqueeze(-1)

    x_pooled = _divide_no_nan(
        torch.sum(x_unpooled * input_mask_expand, axis=-2),
        torch.sum(input_mask.float(), axis=-1, keepdims=True),
    )
    # [*B, D]
    return x_pooled


class ProjectionLayer(torch.nn.Module):

  def __init__(self, in_features, out_features):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features, bias=False)

  def forward(self, x):
    return self.linear(x)
