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
"""Torch-TFL op definitions and fake implementations."""
from ai_edge_torch.odml_torch.experimental.torch_tfl import torch_library_utils
import torch

custom_op_with_fake = torch_library_utils.custom_op_with_fake


@custom_op_with_fake("tfl::batch_matmul")
def tfl_batch_matmul(
    x: torch.Tensor, y: torch.Tensor, adj_x: bool = False, adj_y: bool = False
) -> torch.Tensor:
  if x.ndim < 2 or y.ndim < 2:
    raise ValueError("Input tensors must have at least 2 dimensions.")
  if adj_x:
    x = torch.transpose(x, -1, -2)
  if adj_y:
    y = torch.transpose(y, -1, -2)
  return torch.matmul(x, y)
