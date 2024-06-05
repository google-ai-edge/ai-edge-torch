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
# Cache management utilities.

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn


@dataclass
class KVCaches:
  """A utility dataclass for holding Key and Value cache per layer."""

  k_caches: List[torch.Tensor]
  v_caches: List[torch.Tensor]


def build_kv_cache(
    config: Dict[str, Any],
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> KVCaches:
  """Builds `KVCaches` obejct based on model config.

  Args:
      config (Dict[str, Any]): Model's config.
      dtype (torch.dtype, optional): KV buffers data type. Defaults to
        torch.float32.
      device (torch.device, optional): KV buffers device. Defaults to None.

  Returns:
      KVCaches: A zero initialized `KVCaches` holding one entry per layer.
  """
  cache = KVCaches([], [])
  # Shape: [batch_size, kv_cache_max, n_heads, head_dim]
  shape = (1, config.kv_cache_max, config.attn_config.num_query_groups, config.head_dim)
  for _ in range(config.num_layers):
    cache.k_caches.append(torch.zeros(shape, dtype=dtype, device=device))
    cache.v_caches.append(torch.zeros(shape, dtype=dtype, device=device))
  return cache
