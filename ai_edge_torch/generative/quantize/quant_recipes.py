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

"""Helper functions to create common and supported quantization recipes.

These recipes will work with models created with the Edge Generative API only.
Assume Transformer architecture congruent with
ai_edge_torch/generative/layers/model_config.py:ModelConfig.

Typical usage example:

  quant_config = quant_recipes.full_int8_dynamic_recipe()
  edge_model = ai_edge_torch.convert(
      model, (tokens, input_pos), quant_config=quant_config
  )
"""

from typing import Optional
from ai_edge_torch.generative.layers import model_config
from ai_edge_torch.generative.quantize import quant_attrs
from ai_edge_torch.generative.quantize import quant_recipe
from ai_edge_torch.generative.quantize import quant_recipe_utils
from ai_edge_torch.quantize import quant_config


def full_dynamic_recipe(
    mcfg: model_config.ModelConfig | None = None,
    weight_dtype: quant_attrs.Dtype = quant_attrs.Dtype.INT8,
    granularity: quant_attrs.Granularity = quant_attrs.Granularity.CHANNELWISE,
) -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=quant_recipe_utils.create_layer_quant_dynamic(
              weight_dtype, granularity
          ),
          _model_config=mcfg,
      )
  )


def full_weight_only_recipe(
    mcfg: model_config.ModelConfig | None = None,
    weight_dtype: quant_attrs.Dtype = quant_attrs.Dtype.INT8,
    granularity: quant_attrs.Granularity = quant_attrs.Granularity.CHANNELWISE,
) -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=quant_recipe_utils.create_layer_quant_weight_only(
              weight_dtype, granularity
          ),
          _model_config=mcfg,
      )
  )


def full_fp16_recipe(
    mcfg: model_config.ModelConfig | None = None,
) -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=quant_recipe_utils.create_layer_quant_fp16(),
          _model_config=mcfg,
      )
  )
