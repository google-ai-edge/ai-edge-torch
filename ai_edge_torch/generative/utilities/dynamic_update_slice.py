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
# Common utility functions for data loading etc.
from dataclasses import dataclass
import glob
import os
from typing import Callable, Dict, List, Tuple
from typing import Sequence
import torch


# Use torch.library.custom_op to define a new custom operator.
@torch.library.custom_op("ai_edge_torch::dynamic_update_slice", mutates_args=())
def dynamic_update_slice(
    in_tensor: torch.Tensor, update: torch.Tensor, start_index: torch.Tensor
) -> torch.Tensor:
  compare_size = torch.tensor(in_tensor.size()) == torch.tensor(update.size())
  mismatch = torch.nonzero(~compare_size, as_tuple=False)
  dim = mismatch[0].item() if len(mismatch) > 0 else 0
  start = start_index[dim].item()
  end = start + update.shape[dim]
  indices = torch.arange(start, end).to(torch.long)
  return in_tensor.index_copy(dim, indices, update)


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@dynamic_update_slice.register_fake
def _(in_tensor, update, start_index):
  return in_tensor.clone().detach()
