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

"""Config for customizing model export process."""

import dataclasses
from typing import List, Optional

from absl import flags
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch

dataclass = dataclasses.dataclass


@dataclass
class ExportConfig:
  """Model generating configuration settings."""

  # On prefill signatures, should the model produce logit output?
  # When False, only decode signatures will produce output.
  output_logits_on_prefill: bool = False
  # Attention masks given as inputs to the model.
  prefill_mask: Optional[torch.Tensor | List[torch.Tensor]] = None
  decode_mask: Optional[torch.Tensor | List[torch.Tensor]] = None
  # The KV Cache layout for K and V buffers in attention.
  kvcache_layout: kv_utils.KVLayout = kv_utils.KV_LAYOUT_DEFAULT
  # TODO(b/409373223): The KV Cache class for K and V buffers in attention.
  kvcache_cls: type = kv_utils.KVCache
  # The batch size of the decode signature.
  decode_batch_size: int = 1


def _build_mask(mask_len, kv_cache_max_len) -> torch.Tensor:
  if isinstance(mask_len, list):
    return [_build_mask(i, kv_cache_max_len) for i in mask_len]

  mask = torch.full(
      (mask_len, kv_cache_max_len), float('-inf'), dtype=torch.float32
  )
  mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
  return mask


def get_from_flags() -> ExportConfig:
  """Builds an export config according to the commandline flags."""
  export_config = ExportConfig()

  if flags.FLAGS.mask_as_input:
    export_config.prefill_mask = _build_mask(
        flags.FLAGS.prefill_seq_lens, flags.FLAGS.kv_cache_max_len
    )
    export_config.decode_mask = _build_mask(1, flags.FLAGS.kv_cache_max_len)

  if flags.FLAGS.transpose_kv_cache:
    export_config.kvcache_layout = kv_utils.KV_LAYOUT_TRANSPOSED

  return export_config
