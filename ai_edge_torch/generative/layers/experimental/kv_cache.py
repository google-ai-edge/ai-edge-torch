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

from typing import Any, Dict, List, Tuple

import torch


class KVCache:
  """A utility clss for holding Key and Value caches per layer."""

  def __init__(
      self,
      num_layers: int,
      shape: Tuple[int],
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
  ) -> None:
    """Initialize an instance of KV Cache based on provided arguments.

    Args:
        num_layers (int): The number layers to build the cache for.
        shape (Tuple[int]): The shape of cache. Shared accross the layers.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.
    """
    self.k_caches = []
    self.v_caches = []
    for _ in range(num_layers):
      self.k_caches.append(torch.zeros(shape, dtype=dtype, device=device))
      self.v_caches.append(torch.zeros(shape, dtype=dtype, device=device))

  def get_flattened_tensors(self) -> List[torch.Tensor]:
    """Get interleaved and flattened K and V tensors.

    Returns:
        List[torch.Tensor]: Flattened K and V tensors as
          `[k_0, v_0, k_1, v_1, ...]`.
    """
    tensors = []
    for i in range(len(self.k_caches)):
      tensors.append(self.k_caches[i])
      tensors.append(self.v_caches[i])
    return tensors

  def update_from_flattened_tensors(self, tensors: List[torch.Tensor]):
    """Update caches based on provided flat cache list.

    Args:
        tensors (List[torch.Tensor]): Updated cache tensor provided as a flat
          list of `[k_0, v_0, k_1, v_1, ...]`.
    """
    assert len(tensors) == 2 * len(self.k_caches)
    for i in range(len(self.k_caches)):
      self.k_caches[i] = tensors[2 * i]
      self.v_caches[i] = tensors[2 * i + 1]

  @classmethod
  def from_model_config(
      cls,
      config: Dict[str, Any],
      dtype: torch.dtype = torch.float32,
      device: torch.device = None,
  ) -> "KVCache":
    """Build an instance of the class based on model config.

    Args:
        config (Dict[str, Any]): Model config used for building the cache.
        dtype (torch.dtype, optional): The data type of the cache tensor.
          Defaults to torch.float32.
        device (torch.device, optional): The device placement of the cache
          tensors. Defaults to None.

    Returns:
        KVCache: The created cache object.
    """
    shape = (
        1,
        config.kv_cache_max,
        config.attn_config.num_query_groups,
        config.head_dim,
    )
    obj = cls(config.num_layers, shape, dtype=dtype, device=device)
    return obj
