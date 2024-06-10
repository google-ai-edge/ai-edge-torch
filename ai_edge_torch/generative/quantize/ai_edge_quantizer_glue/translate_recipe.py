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

import json

from ai_edge_quantizer import quantizer

from ai_edge_torch.generative.quantize import quant_attrs
from ai_edge_torch.generative.quantize import quant_recipe

_OpExecutionMode = quantizer.qtyping.OpExecutionMode
_OpName = quantizer.qtyping.TFLOperationName
_TensorQuantConfig = quantizer.qtyping.TensorQuantizationConfig
_OpQuantConfig = quantizer.qtyping.OpQuantizationConfig

_DEFAULT_REGEX_STR = '.*'
_ATTENTION_IDX_REGEX_STR = (
    'transformer_blocks\[{}\]/ai_edge_torch.generative.layers.attention'
)
_FEEDFORWARD_IDX_REGEX_STR = (
    'transformer_blocks\[{}\]/ai_edge_torch.generative.layers.feed_forward'
)
_EMBEDDING_REGEX_STR = 'Embedding_tok_embedding'
_ANY_TWO_DIGITS_REGEX_STR = '\d{1,2}'


def _get_nbits_from_dtype(dtype: quant_attrs.Dtype) -> int:
  if dtype == quant_attrs.Dtype.FP32:
    return 32
  elif dtype == quant_attrs.Dtype.FP16:
    return 16
  elif dtype == quant_attrs.Dtype.INT8:
    return 8
  raise ValueError('Unimplemented number of bits')


def _get_dtype_from_dtype(dtype: quant_attrs.Dtype) -> quantizer.qtyping.TensorDataType:
  if dtype == quant_attrs.Dtype.FP32 or dtype == quant_attrs.Dtype.FP16:
    return quantizer.qtyping.TensorDataType.FLOAT
  else:
    return quantizer.qtyping.TensorDataType.INT


def _get_execution_mode_from_mode(mode: quant_attrs.Mode) -> _OpExecutionMode:
  if mode == quant_attrs.Mode.DYNAMIC_RANGE:
    return _OpExecutionMode.DRQ
  elif mode == quant_attrs.Mode.WEIGHT_ONLY:
    return _OpExecutionMode.WEIGHT_ONLY
  raise ValueError('Unimplemented execution mode')


def _get_channelwise_from_granularity(granularity: quant_attrs.Granularity) -> bool:
  if granularity == quant_attrs.Granularity.CHANNELWISE:
    return True
  elif granularity == quant_attrs.Granularity.NONE:
    return False
  raise ValueError('Unimplemented granularity')


def _get_algorithm_key_from_algorithm(algo: quant_attrs.Algorithm) -> str:
  if algo == quant_attrs.Algorithm.MIN_MAX:
    return quantizer.algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT
  elif algo == quant_attrs.Algorithm.FLOAT_CAST:
    return quantizer.algorithm_manager.AlgorithmName.FLOAT_CASTING
  raise ValueError('Unimplemented algorithm')


def _set_quant_config(
    rm: quantizer.recipe_manager.RecipeManager,
    layer_recipe: quant_recipe.LayerQuantRecipe,
    regex: str,
):
  support_op_list = [_OpName.FULLY_CONNECTED, _OpName.CONV_2D]
  if layer_recipe.algorithm == quant_attrs.Algorithm.MIN_MAX:
    support_op_list += [_OpName.BATCH_MATMUL, _OpName.EMBEDDING_LOOKUP]
  for op_name in support_op_list:
    rm.add_quantization_config(
        regex=regex,
        operation_name=op_name,
        op_config=_OpQuantConfig(
            weight_tensor_config=_TensorQuantConfig(
                num_bits=_get_nbits_from_dtype(layer_recipe.weight_dtype),
                symmetric=True,
                channel_wise=_get_channelwise_from_granularity(
                    layer_recipe.granularity
                ),
                dtype=_get_dtype_from_dtype(layer_recipe.weight_dtype),
            ),
            execution_mode=_get_execution_mode_from_mode(layer_recipe.mode),
        ),
        algorithm_key=_get_algorithm_key_from_algorithm(layer_recipe.algorithm),
        override_algorithm=True,
    )


def translate_to_ai_edge_recipe(
    recipe: quant_recipe.GenerativeQuantRecipe,
) -> quantizer.recipe_manager.ModelQuantizationRecipe:
  rm = quantizer.recipe_manager.RecipeManager()

  if recipe.default is not None:
    _set_quant_config(rm, recipe.default, _DEFAULT_REGEX_STR)

  if recipe.embedding is not None:
    _set_quant_config(rm, recipe.embedding, _EMBEDDING_REGEX_STR)

  if recipe.attention is not None:
    if isinstance(recipe.attention, dict):
      for idx, layer in recipe.attention.items():
        _set_quant_config(rm, layer, _ATTENTION_IDX_REGEX_STR.format(idx))
    else:
      _set_quant_config(
          rm,
          recipe.attention,
          _ATTENTION_IDX_REGEX_STR.format(_ANY_TWO_DIGITS_REGEX_STR),
      )

  if recipe.feedforward is not None:
    if isinstance(recipe.feedforward, dict):
      for idx, layer in recipe.feedforward.items():
        _set_quant_config(rm, layer, _FEEDFORWARD_IDX_REGEX_STR.format(idx))
    else:
      _set_quant_config(
          rm,
          recipe.feedforward,
          _FEEDFORWARD_IDX_REGEX_STR.format(_ANY_TWO_DIGITS_REGEX_STR),
      )

  return rm.get_quantization_recipe()


def quantize_model(
    model: bytearray, recipe: quantizer.recipe_manager.ModelQuantizationRecipe
) -> bytearray:
  # TODO(b/336599483): Remove tempfile and use bytearray instead
  tmp_model_path = '/tmp/tmp.tflite'
  tmp_recipe_path = '/tmp/recipe.json'
  with open(tmp_model_path, 'wb') as fp:
    fp.write(model)
  with open(tmp_recipe_path, 'w') as rp:
    rp.write(json.dumps(recipe))

  qt = quantizer.Quantizer(tmp_model_path, tmp_recipe_path)
  result = qt.quantize()

  # TODO(b/336599483): Remove tempfile and use bytearray instead
  import os

  os.remove(tmp_model_path)
  os.remove(tmp_recipe_path)

  return result.quantized_model
