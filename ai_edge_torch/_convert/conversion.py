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

import logging
import os
from typing import Any, Optional

import torch

from ai_edge_torch import lowertools, model
from ai_edge_torch._convert import fx_passes, signature
from ai_edge_torch.generative import fx_passes as generative_fx_passes
from ai_edge_torch.quantize import quant_config as qcfg

os.environ["EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM"] = "1"


def run_convert_passes(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
  exported_program = generative_fx_passes.run_generative_passes(
      exported_program
  )
  return fx_passes.run_passes(
      exported_program,
      [
          fx_passes.BuildInterpolateCompositePass(),
          fx_passes.CanonicalizePass(),
          fx_passes.OptimizeLayoutTransposesPass(),
          fx_passes.CanonicalizePass(),
          fx_passes.BuildAtenCompositePass(),
          fx_passes.CanonicalizePass(),
          fx_passes.InjectMlirDebuginfoPass(),
          fx_passes.CanonicalizePass(),
      ],
  )


def _warn_training_modules(
    module: torch.nn.Module,
    signature: signature.Signature,
):
  """Warns the user if the module is in training mode (.eval not called)."""
  if not module.training:
    return

  message = (
      "Your model {sig_name}is converted in training mode. Please set the"
      " module in evaluation mode with `module.eval()` for better on-device"
      " performance and compatibility."
  )
  if signature.name == model.DEFAULT_SIGNATURE_NAME:
    # User does not specify any signature names explicitly.
    message = message.format(sig_name="")
  else:
    message = message.format(sig_name=f'"{signature.name}" ')

  logging.warning(message)


def convert_signatures(
    signatures: list[signature.Signature],
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: Optional[dict[str, Any]],
) -> model.TfLiteModel:
  """Converts a list of `signature.Signature`s and embeds them into one `model.TfLiteModel`.

  Args:
      signatures: The list of 'signature.Signature' objects containing PyTorch
        modules to be converted.
      quant_config: User-defined quantization method and scheme of the model.
      _tfl_converter_flags: A nested dictionary allowing setting flags for the
        underlying tflite converter.

  Returns:
    The converted `model.TfLiteModel` object.
  """
  if _tfl_converter_flags is None:
    _tfl_converter_flags = {}

  # Apply default fx passes
  tflite_model = lowertools.exported_programs_to_tflite(
      signatures,
      quant_config=quant_config,
      _tfl_converter_flags=_tfl_converter_flags,
  )

  return model.TfLiteModel(tflite_model)
