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

from .convert.converter import convert
from .convert.converter import signature
from .convert.to_channel_last_io import to_channel_last_io
from .model import Model


def load(path: str) -> Model:
  """Imports an ai_edge_torch model from disk.

  Args:
    path: The path to the serialized ai_edge_torch model.

  Returns:
    An ai_edge_torch.model.Model object.
  """
  return Model.load(path)
