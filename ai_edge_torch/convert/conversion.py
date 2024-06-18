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

import gc
import logging
import os
from typing import Optional

import torch
from torch.export import ExportedProgram
from torch_xla import stablehlo

from ai_edge_torch import model
from ai_edge_torch.convert import conversion_utils as cutils
from ai_edge_torch.convert.fx_passes import BuildAtenCompositePass
from ai_edge_torch.convert.fx_passes import BuildInterpolateCompositePass  # NOQA
from ai_edge_torch.convert.fx_passes import CanonicalizePass
from ai_edge_torch.convert.fx_passes import InjectMlirDebuginfoPass
from ai_edge_torch.convert.fx_passes import OptimizeLayoutTransposesPass
from ai_edge_torch.convert.fx_passes import run_passes
from ai_edge_torch.generative.fx_passes import run_generative_passes
from ai_edge_torch.quantize import quant_config as qcfg

os.environ["EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM"] = "1"


def _run_convert_passes(
    exported_program: ExportedProgram,
) -> ExportedProgram:
  exported_program = run_generative_passes(exported_program)
  return run_passes(
      exported_program,
      [
          BuildInterpolateCompositePass(),
          CanonicalizePass(),
          OptimizeLayoutTransposesPass(),
          CanonicalizePass(),
          BuildAtenCompositePass(),
          CanonicalizePass(),
          InjectMlirDebuginfoPass(),
          CanonicalizePass(),
      ],
  )


def _warn_training_modules(signatures: list[cutils.Signature]):
  for sig in signatures:
    if not sig.module.training:
      continue

    message = (
        "Your model {sig_name}is converted in training mode. "
        "Please set the module in evaluation mode with `module.eval()` for better on-device performance and compatibility."
    )
    if len(signatures) == 1 and sig.name == cutils.DEFAULT_SIGNATURE_NAME:
      # User does not specify any signature names explicitly.
      message = message.format(sig_name="")
    else:
      message = message.format(sig_name=f'"{sig.name}" ')

    logging.warn(message)


def convert_signatures(
    signatures: list[cutils.Signature],
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: dict = {},
) -> model.TfLiteModel:
  """Converts a list of `Signature`s and embeds them into one `model.TfLiteModel`.
  Args:
      signatures: The list of 'Signature' objects containing PyTorch modules to be converted.
      quant_config: User-defined quantization method and scheme of the model.
      _tfl_converter_flags: A nested dictionary allowing setting flags for the underlying tflite converter.
  """
  _warn_training_modules(signatures)

  exported_programs: torch.export.ExportedProgram = [
      torch.export.export(
          sig.module, sig.sample_args, dynamic_shapes=sig.dynamic_shapes
      )
      for sig in signatures
  ]

  # Apply default fx passes
  exported_programs = list(map(_run_convert_passes, exported_programs))
  shlo_bundles: list[stablehlo.StableHLOModelBundle] = [
      cutils.exported_program_to_stablehlo_bundle(exported, sig.sample_args)
      for exported, sig in zip(exported_programs, signatures)
  ]

  merged_shlo_graph_module: stablehlo.StableHLOGraphModule = (
      cutils.merge_stablehlo_bundles(shlo_bundles, signatures, exported_programs)
  )
  del exported_programs
  del shlo_bundles

  gc.collect()

  tflite_model = cutils.convert_stablehlo_to_tflite(
      merged_shlo_graph_module,
      signatures,
      quant_config=quant_config,
      _tfl_converter_flags=_tfl_converter_flags,
  )

  return model.TfLiteModel(tflite_model)
