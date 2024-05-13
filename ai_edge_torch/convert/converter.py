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

from typing import Any, Dict, Optional, Tuple, Union

import torch

from ai_edge_torch import model
from ai_edge_torch.convert import conversion
from ai_edge_torch.convert import conversion_utils as cutils
from ai_edge_torch.quantize import quant_config as qcfg


class Converter:

  def __init__(self):
    self._signatures: list[cutils.Signature] = []

  def signature(
      self,
      name: str,
      module: torch.nn.Module,
      sample_args: tuple[cutils.TracingArg],
      dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
  ) -> Converter:
    """Alias to `add_signature`"""
    return self.add_signature(name, module, sample_args, dynamic_shapes)

  def add_signature(
      self,
      name: str,
      module: torch.nn.Module,
      sample_args: tuple[cutils.TracingArg],
      dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
  ) -> Converter:
    """Allows adding a new named torch model along with sample args to the conversion.

    Args:
      name: The name of the signature included in the converted edge model.
      module: The torch module to be converted.
      sample_args: Tuple of args by which the torch module will be traced prior to conversion.
      dynamic_shapes: Optional dict or tuple that specify dynamic shape specifications for each input in original order.
        See https://pytorch.org/docs/stable/export.html#expressing-dynamism for more details.

    Raises:
      ValueError: If a signature with the provided name already exists.
    """

    if name in [sig.name for sig in self._signatures]:
      raise ValueError(f"A signature with the provided name ({name}) is already added.")

    self._signatures.append(cutils.Signature(name, module, sample_args, dynamic_shapes))
    return self

  def convert(
      self,
      module: torch.nn.Module = None,
      sample_args: tuple[cutils.TracingArg] = None,
      *,
      quant_config: Optional[qcfg.QuantConfig] = None,
      dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
      _ai_edge_converter_flags: dict = {},
  ) -> model.TfLiteModel:
    """Finalizes the conversion and produces an edge model.

    This could be called with no arguments as follows:

      edge_model = Converter().signature(name, module, args).convert()

    Or it could be used to set the default signature for the converted edge model:

      edge_model =  Converter().convert(module, args)

    Args:
      name: The name of the signature included in the converted edge model.
      module: The torch module to be converted.
      sample_args: Tuple of args by which the torch module will be traced prior to conversion.
      quant_config: User-defined quantization method and scheme of the model.
      dynamic_shapes: Optional dict or tuple that specify dynamic shape specifications for each input in original order.
        See https://pytorch.org/docs/stable/export.html#expressing-dynamism for more details.
      _ai_edge_converter_flags: A nested dictionary allowing setting flags for the underlying converter.
        This gives access to an implementation detail of this function and so needs to be treated as such.
        Please do not rely on this parameter except for local debugging as this can be removed in a future release.

    Raises:
      ValueError: If the arguments are not provided as expected. See the example in this functions's comment.
    """
    if module is not None:
      if sample_args is not None:  # both module and args provided
        self.add_signature(
            cutils.DEFAULT_SIGNATURE_NAME, module, sample_args, dynamic_shapes
        )
      else:  # module is provided but not sample_args
        raise ValueError("sample_args needs to be provided if a module is specified.")

    return conversion.convert_signatures(
        self._signatures,
        quant_config=quant_config,
        _tfl_converter_flags=_ai_edge_converter_flags,
    )


def signature(
    name: str,
    module: torch.nn.Module,
    sample_args: tuple[cutils.TracingArg],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> Converter:
  """Initiates a Converter object with the provided signature.

  Args:
    name: The name of the signature included in the converted edge model.
    module: The torch module to be converted.
    sample_args: Tuple of args by which the torch module will be traced prior to conversion.
    dynamic_shapes: Optional dict or tuple that specify dynamic shape specifications for each input in original order.
      See https://pytorch.org/docs/stable/export.html#expressing-dynamism for more details.

  Example:
    converter = ai_edge_torch.signature(name, module, args)
    edge_model = converter.convert()

  """
  return Converter().signature(name, module, sample_args, dynamic_shapes)


def convert(
    module: torch.nn.Module = None,
    sample_args: tuple[cutils.TracingArg] = None,
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    _ai_edge_converter_flags: dict = {},
) -> model.TfLiteModel:
  """Allows converting a PyTorch model to an edge model with one default signature in one step.

  Args:
    module: The torch module to be converted.
    sample_args: Tuple of args by which the torch module will be traced prior to conversion.
    quant_config: User-defined quantization method and scheme of the model.
    dynamic_shapes: Optional dict or tuple that specify dynamic shape specifications for each input in original order.
      See https://pytorch.org/docs/stable/export.html#expressing-dynamism for more details.
    _ai_edge_converter_flags: A nested dictionary allowing setting flags for the underlying converter.
      This gives access to an implementation detail of this function and so needs to be treated as such.
      Please do not rely on this parameter except for local debugging as this can be removed in a future release.

  Example:
    edge_model = ai_edge_torch.convert(module, args)

  """

  return Converter().convert(
      module,
      sample_args,
      quant_config=quant_config,
      dynamic_shapes=dynamic_shapes,
      _ai_edge_converter_flags=_ai_edge_converter_flags,
  )
