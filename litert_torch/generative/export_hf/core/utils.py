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
"""Utility functions."""

import torch


# TODO(weiyiw): Remove this when we support external sliding mask.
def create_sliding_mask(
    segment_pos: torch.Tensor,  # [B, L]
    cache_len: int,
    sliding_window_size: int,
    mask_value: float = -1e4,
) -> torch.Tensor:
  """Creates mask for sliding window attention (PyTorch)."""
  # Use torch.arange to create a tensor with a range of integers in a
  # Dynamo-friendly way.
  cache_positions = torch.arange(cache_len, dtype=torch.int32)
  cache_positions = cache_positions.view(1, 1, -1)  # [1, 1, cache_len]
  segment_pos_expanded = segment_pos.clone().unsqueeze(-1)  # [B, seq_len, 1]

  # Create boolean masks for window boundaries.
  left_boundary = cache_positions > segment_pos_expanded - sliding_window_size
  right_boundary = cache_positions < segment_pos_expanded + sliding_window_size

  # Combine boolean masks (AND).
  sliding_mask_bool = left_boundary & right_boundary

  # Convert boolean mask to float mask with 0 and -inf.
  sliding_mask = torch.where(
      sliding_mask_bool,
      torch.zeros_like(sliding_mask_bool, dtype=torch.float),
      torch.full_like(sliding_mask_bool, mask_value, dtype=torch.float),
  )

  return sliding_mask.unsqueeze(0)


WARNING_MESSAGE = r"""
                   .
                  / \
                 / ! \
                /  .  \
               /_______\
              / WARNING \
             /___________\
            /_____________\

"""


ERROR_MESSAGE = r"""
                   .
                  / \
                 / ! \
                /  .  \
               /_______\
              /  ERROR  \
             /___________\
            /_____________\
           /_______________\

"""


def has_local_rope(model):
  if hasattr(model, 'language_model'):
    model = model.language_model
  return hasattr(model.model, 'rotary_emb_local')


def has_sliding_attention(model):
  if hasattr(model, 'language_model'):
    model = model.language_model
  sliding_window = getattr(model.config, 'sliding_window', None)
  if not sliding_window:
    return False
  layer_types = getattr(model.config, 'layer_types', None)
  return layer_types is not None and 'sliding_attention' in layer_types
