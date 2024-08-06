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

from __future__ import annotations

import copy
import functools
from typing import Any, Callable, Dict, List, Optional, Set

from ai_edge_torch.quantize.pt2e_quantizer_utils import _convert_scalars_to_attrs  # NOQA
from ai_edge_torch.quantize.pt2e_quantizer_utils import OP_TO_ANNOTATOR
from ai_edge_torch.quantize.pt2e_quantizer_utils import OperatorConfig
from ai_edge_torch.quantize.pt2e_quantizer_utils import OperatorPatternType
from ai_edge_torch.quantize.pt2e_quantizer_utils import propagate_annotation
from ai_edge_torch.quantize.pt2e_quantizer_utils import QuantizationConfig
import torch
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization.observer import MovingAveragePerChannelMinMaxObserver  # NOQA
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.observer import PlaceholderObserver
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.ao.quantization.quantizer import FixedQParamsQuantizationSpec
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer import Quantizer
from torch.fx import Node
import torch.nn.functional as F

__all__ = [
    "PT2EQuantizer",
    "get_symmetric_quantization_config",
]


def _supported_symmetric_quantized_operators() -> (
    Dict[str, List[OperatorPatternType]]
):
  supported_operators: Dict[str, List[OperatorPatternType]] = {
      # Both conv and linear should be able to handle relu + hardtanh fusion since
      # those are clamp ops
      "conv2d": [
          [torch.nn.Conv2d, torch.nn.ReLU],
          [torch.nn.Conv2d, F.relu],
          [F.conv2d, torch.nn.ReLU],
          [F.conv2d, F.relu],
      ],
      "linear": [[torch.nn.Linear], [F.linear]],
      "add": [[torch.add]],
      "max_pool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
      "adaptive_avg_pool2d": [
          [torch.nn.AdaptiveAvgPool2d],
          [F.adaptive_avg_pool2d],
      ],
  }
  return copy.deepcopy(supported_operators)


def _get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
  supported_config_and_operators: List[OperatorConfig] = []
  for quantization_config in [
      get_symmetric_quantization_config(),
      get_symmetric_quantization_config(is_qat=True),
      get_symmetric_quantization_config(is_per_channel=True),
      get_symmetric_quantization_config(is_per_channel=True, is_qat=True),
  ]:
    ops = _supported_symmetric_quantized_operators()
    for pattern_list in ops.values():
      supported_config_and_operators.append(
          OperatorConfig(quantization_config, pattern_list)
      )
  return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_symmetric_quantization_config(
    is_per_channel: bool = False,
    is_qat: bool = False,
    is_dynamic: bool = False,
):
  if is_qat:
    if is_dynamic:
      raise NotImplementedError(
          "dynamic quantization for qat is not yet implemented."
      )
    act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
  else:
    if is_dynamic:
      act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
    else:
      act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

  act_quantization_spec = QuantizationSpec(
      dtype=torch.int8,
      quant_min=-128,
      quant_max=127,
      qscheme=torch.per_tensor_affine,
      is_dynamic=is_dynamic,
      observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
          eps=2**-12
      ),
  )
  qscheme = (
      torch.per_channel_symmetric
      if is_per_channel
      else torch.per_tensor_symmetric
  )
  weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
      MinMaxObserver
  )
  if is_qat:
    weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
  elif is_per_channel:
    weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

  extra_args: Dict[str, Any] = {"eps": 2**-12}
  if is_qat:
    if qscheme == torch.per_tensor_symmetric:
      extra_args["observer"] = MovingAverageMinMaxObserver
    else:
      extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
  weight_quantization_spec = QuantizationSpec(
      dtype=torch.int8,
      quant_min=-127,
      quant_max=127,
      qscheme=qscheme,
      ch_axis=0,
      is_dynamic=False,
      observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
          **extra_args
      ),
  )

  bias_quantization_spec = None

  # Some TFLite ops (e.g. Logistic, Softmax) have fixed qparams requirements
  fixed_qparams_spec = FixedQParamsQuantizationSpec(
      dtype=torch.int8,
      scale=1 / 256,
      zero_point=-128,
      quant_min=-128,
      quant_max=127,
      qscheme=torch.per_tensor_affine,
  )

  if is_dynamic:
    # Only valid for TFLite downstream to have no input activation quantization
    # because dynamic quantization should be legalized to TFLite DRQ kernels
    # which calculate quantization parameters during runtime inside the kernels
    quantization_config = QuantizationConfig(
        None,
        None,
        weight_quantization_spec,
        bias_quantization_spec,
        None,
        is_qat,
        True,
    )
  else:
    quantization_config = QuantizationConfig(
        act_quantization_spec,
        act_quantization_spec,
        weight_quantization_spec,
        bias_quantization_spec,
        fixed_qparams_spec,
        is_qat,
        False,
    )
  return quantization_config


def _get_supported_config_and_operators() -> List[OperatorConfig]:
  return _get_supported_symmetric_config_and_operators()


def _get_module_name_filter(module_name: str):
  """Get the module_name_filter function for a given module name, the filter accepts

  a node and checks if the node comes from a module that has certain module name

  For example:
      node: linear_op = call_function[...](...)  # comes from a module with name
      blocks.sub.linear1


  >> module_name_filter = _get_module_name_filter("blocks.sub")
  >> print(module_name_filter(node))
  True  # the node is from "blocks.sub" based on the fully qualified name
  "blocks.sub.linear1"
  """

  def module_name_filter(n: Node) -> bool:
    # example: {
    #    'L__self___sub': ("L['self'].sub", <class '....Sub'>),
    #    'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
    # }
    # get_attr nodes doesn't have nn_module_stack?
    nn_module_stack = n.meta.get("nn_module_stack", {})
    names = [
        n[len("L__self___") :].replace("_", ".") for n in nn_module_stack.keys()
    ]
    return module_name in names

  return module_name_filter


def _get_module_type_filter(tp: Callable):
  """Get the module_type_filter function for a given module type, the filter accepts

  a node and checks if the node comes from a module that has certain module type

  For example:
      node: linear_op = call_function[...](...)  # comes from a module with type
      Block -> Sub -> Linear


  >> module_type_filter = _get_module_type_filter(Sub)  # submodule with type
  `Sub`, under the `Block` submodule
  >> print(module_type_filter(node))
  True  # the node is from the submodule `Sub` (same for `Block` and `Linear` as
  well)
  """

  def module_type_filter(n: Node) -> bool:
    # example: {
    #     'L__self___sub': ("L['self'].sub", <class '....Sub'>),
    #     'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
    # }
    nn_module_stack = n.meta.get("nn_module_stack", {})
    types = [t for _, t in nn_module_stack.values()]
    return tp in types

  return module_type_filter


def _get_not_module_type_or_name_filter(
    tp_list: List[Callable], module_name_list: List[str]
) -> Callable[[Node], bool]:
  module_type_filters = [_get_module_type_filter(tp) for tp in tp_list]
  module_name_list_filters = [
      _get_module_name_filter(m) for m in module_name_list
  ]

  def not_module_type_or_name_filter(n: Node) -> bool:
    return not any(f(n) for f in module_type_filters + module_name_list_filters)

  return not_module_type_or_name_filter


class PT2EQuantizer(Quantizer):
  supported_config_and_operators = _get_supported_config_and_operators()
  STATIC_QAT_ONLY_OPS = [
      "conv_bn_relu",
      "conv_bn",
  ]

  # static quantization ops (both PTQ and QAT)
  STATIC_OPS = [
      "linear",
      "addmm",
      "conv_relu",
      "conv",
      "adaptive_avg_pool2d",
      "gru_io_only",
      "max_pool2d",
      "add_relu",
      "add",
      "mul_relu",
      "mul",
      "cat",
      "fixed_qparams",
  ]

  DYNAMIC_OPS = [
      "linear",
      "addmm",
      "conv",
      "conv_relu",
  ]

  def __init__(self):
    super().__init__()
    self.global_config: Optional[QuantizationConfig] = None
    self.operator_type_config: Dict[
        torch._ops.OpOverloadPacket, Optional[QuantizationConfig]
    ] = {}
    self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
    self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

  @classmethod
  def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
    op_configs: Set[QuantizationConfig] = set({})
    for spec, _ in cls.supported_config_and_operators:
      op_configs.add(spec)
    return list(op_configs)

  @classmethod
  def get_supported_operator_for_quantization_config(
      cls, quantization_config: Optional[QuantizationConfig]
  ) -> List[OperatorPatternType]:
    if quantization_config is None:
      all_ops = []
      for _, ops in cls.supported_config_and_operators:
        all_ops.extend(ops)
      return all_ops

    for config, ops in cls.supported_config_and_operators:
      # note: this assumes each entry in cls.supported_spec_and_operators
      # corresponds to one spec, e.g. we don't have
      # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
      # where the first and second entry have the same spec but did not
      # merge the op list
      if config == quantization_config:
        return ops
    return []

  def set_global(
      self, quantization_config: QuantizationConfig
  ) -> PT2EQuantizer:
    self.global_config = quantization_config
    return self

  def set_operator_type(
      self,
      operator_type: torch._ops.OpOverloadPacket,
      quantization_config: QuantizationConfig,
  ) -> PT2EQuantizer:
    self.operator_type_config[operator_type] = quantization_config
    return self

  def set_module_type(
      self, module_type: Callable, quantization_config: QuantizationConfig
  ):
    """Set quantization_config for a submodule with type: `module_type`, for example:

    quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it
    will quantize all supported operator/operator
    patterns in the submodule with this module type with the given
    `quantization_config`
    """
    self.module_type_config[module_type] = quantization_config
    return self

  def set_module_name(
      self, module_name: str, quantization_config: Optional[QuantizationConfig]
  ):
    """Set quantization_config for a submodule with name: `module_name`, for example:

    quantizer.set_module_name("blocks.sub"), it will quantize all supported
    operator/operator
    patterns in the submodule with this module name with the given
    `quantization_config`
    """
    assert (
        quantization_config is not None
    ), " quantization_config == None is not supported yet"
    self.module_name_config[module_name] = quantization_config
    return self

  def transform_for_annotation(
      self, model: torch.fx.GraphModule
  ) -> torch.fx.GraphModule:
    """Transforms scalar values to tensor attributes"""
    return _convert_scalars_to_attrs(model)

  def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """just handling global spec for now"""
    if self.global_config and not self.global_config.input_activation:  # type: ignore[union-attr]
      model = self._annotate_for_dynamic_quantization_config(model)
    else:
      model = self._annotate_for_static_quantization_config(model)
    propagate_annotation(model)
    return model

  def _annotate_all_static_patterns(
      self,
      model: torch.fx.GraphModule,
      quantization_config: Optional[QuantizationConfig],
      filter_fn: Optional[Callable[[Node], bool]] = None,
  ) -> torch.fx.GraphModule:
    if quantization_config is None:
      return model

    if quantization_config.is_qat:
      for op in self.STATIC_QAT_ONLY_OPS:
        OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
    for op in self.STATIC_OPS:
      OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
    return model

  def _annotate_all_dynamic_patterns(
      self,
      model: torch.fx.GraphModule,
      quantization_config: Optional[QuantizationConfig],
      filter_fn: Optional[Callable[[Node], bool]] = None,
  ) -> torch.fx.GraphModule:
    if quantization_config is None:
      return model

    for op in self.DYNAMIC_OPS:
      OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
    return model

  def _annotate_for_static_quantization_config(
      self, model: torch.fx.GraphModule
  ) -> torch.fx.GraphModule:
    module_name_list = list(self.module_name_config.keys())
    for module_name, config in self.module_name_config.items():
      self._annotate_all_static_patterns(
          model, config, _get_module_name_filter(module_name)
      )

    tp_list = list(self.module_type_config.keys())
    for module_type, config in self.module_type_config.items():
      self._annotate_all_static_patterns(
          model, config, _get_module_type_filter(module_type)
      )

    self._annotate_all_static_patterns(
        model,
        self.global_config,
        _get_not_module_type_or_name_filter(tp_list, module_name_list),
    )
    return model

  def _annotate_for_dynamic_quantization_config(
      self, model: torch.fx.GraphModule
  ) -> torch.fx.GraphModule:
    module_name_list = list(self.module_name_config.keys())
    for module_name, config in self.module_name_config.items():
      self._annotate_all_dynamic_patterns(
          model, config, _get_module_name_filter(module_name)
      )

    tp_list = list(self.module_type_config.keys())
    for module_type, config in self.module_type_config.items():
      self._annotate_all_dynamic_patterns(
          model, config, _get_module_type_filter(module_type)
      )

    self._annotate_all_dynamic_patterns(
        model,
        self.global_config,
        _get_not_module_type_or_name_filter(tp_list, module_name_list),
    )
    return model

  def validate(self, model: torch.fx.GraphModule) -> None:
    pass

  @classmethod
  def get_supported_operators(cls) -> List[OperatorConfig]:
    return cls.supported_config_and_operators
