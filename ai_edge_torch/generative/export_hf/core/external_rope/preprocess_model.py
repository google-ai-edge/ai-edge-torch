# Copyright 2026 The AI Edge Torch Authors.
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
"""Preprocess model for externalized rotary embedding."""

from typing import Any

from ai_edge_torch.generative.export_hf.core import utils
import torch


class RotaryPositionEmbeddingInjector(torch.nn.Module):
  """Injects RotaryPositionEmbedding."""

  data: Any = None

  def __init__(self, config):
    super().__init__()
    self.config = config

  def forward(self, x, position_ids):
    del x, position_ids

    assert self.data is not None
    return self.data


def inject_rotary_position_embedding(model):
  """Injects RotaryPositionEmbedding."""
  if hasattr(model, 'language_model'):
    raise ValueError(
        'inject_rotary_position_embedding should be called on the text model.'
    )
  model.model.original_rotary_emb = model.model.rotary_emb
  model.model.rotary_emb = RotaryPositionEmbeddingInjector(model.model.config)

  if utils.has_local_rope(model):
    model.model.original_rotary_emb_local = model.model.rotary_emb_local
    model.model.rotary_emb_local = RotaryPositionEmbeddingInjector(
        model.model.config
    )
  return model
