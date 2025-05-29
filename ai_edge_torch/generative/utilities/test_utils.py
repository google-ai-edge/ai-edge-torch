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
"""Test utils for generative layers."""
from typing import Sequence
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch


def initialize_kv_cache_all_zeros(
    kv_shape: Sequence[int],
    layout: kv_utils.KVLayout = kv_utils.KV_LAYOUT_DEFAULT,
) -> kv_utils.KVCacheEntry:
  return kv_utils.KVCacheEntry(
      k_cache=torch.zeros(kv_shape, dtype=torch.float32),
      v_cache=torch.zeros(kv_shape, dtype=torch.float32),
      kv_layout=layout,
  )
