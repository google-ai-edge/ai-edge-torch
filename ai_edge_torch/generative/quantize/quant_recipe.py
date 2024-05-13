# Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.
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

from dataclasses import dataclass
import enum
from typing import Optional

from ai_edge_torch.generative.quantize import quant_attrs
from ai_edge_torch.generative.quantize import supported_schemes


@dataclass
class LayerQuantRecipe:
  """Quantization recipe for a single Edge Generative API layer (e.g. Attention).

  Generic layer-scoped quantization recipe that specifies how this layer should
  be quantized by the Edge Generative API. This is applicable to layers implemented
  in ai_edge_torch/generative/layers/. Combinations of attributes that are not
  supported during runtime will be detected when .verify() is called.

  Attributes:
    activation_dtype: Desired data type of activation tensors.
    weight_dtype: Desired data type of weight tensors.
    mode: Type of quantization.
    algorithm: Algorithm for calculating quantization parameters.
    granularity: Granularity of quantization.
  """

  activation_dtype: quant_attrs.Dtype
  weight_dtype: quant_attrs.Dtype
  mode: quant_attrs.Mode
  algorithm: quant_attrs.Algorithm
  granularity: quant_attrs.Granularity

  def __str__(self):
    return (
        f'(a:{self.activation_dtype.name}, '
        f'w:{self.weight_dtype.name}, '
        f'{self.mode.name}, '
        f'{self.algorithm.name}, '
        f'{self.granularity.name})'
    )

  __repr__ = __str__

  def verify(self):
    """Checks if all attributes configured are supported in runtime.

    Raises:
      ValueError: If any attributes are incompatible.
    """
    is_valid = False
    for supported in supported_schemes.get_supported_layer_schemes():
      if (
          self.activation_dtype == supported[0]
          and self.weight_dtype == supported[1]
          and self.mode == supported[2]
          and self.algorithm == supported[3]
          and self.granularity == supported[4]
      ):
        is_valid = True
        break

    if not is_valid:
      raise ValueError(
          'Unsupported LayerQuantRecipe configuration. See get_supported_recipe_matrix()'
      )


@dataclass
class TransformerQuantRecipe:
  """Quantization recipe for a model composed of the Edge Generative API layers.

  Attributes:
    default: The quantization recipe for global scope of the model.
  """

  default: Optional[LayerQuantRecipe] = None

  def __str__(self):
    return f"""TransformerQuantRecipe(
  Default: {self.default}
)"""

  __repr__ = __str__

  def verify(self):
    """Checks if the recipe configured can be supported in runtime.

    Raises:
      ValueError: If the recipe configured is invalid or unsupported.
    """
    if self.default is not None:
      self.default.verify()
