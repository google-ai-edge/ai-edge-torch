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
from typing import Any, Literal, Optional, Union

import ai_edge_torch
from ai_edge_torch import fx_infra
from ai_edge_torch import lowertools
from ai_edge_torch import model
from ai_edge_torch._convert import fx_passes
from ai_edge_torch._convert import signature
from ai_edge_torch.generative import fx_passes as generative_fx_passes
from ai_edge_torch.quantize import quant_config as qcfg
import torch

from ai_edge_litert.aot import aot_compile as aot_compile_lib
from ai_edge_litert.aot.core import types as litert_types


def _run_convert_passes(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
  exported_program = generative_fx_passes.run_generative_passes(
      exported_program
  )

  passes = [
      fx_passes.EliminateDeadCodePass(),
      fx_passes.OptimizeLayoutTransposesPass(),
      fx_passes.CanonicalizePass(),
      fx_passes.BuildAtenCompositePass(),
      fx_passes.RemoveNonUserOutputsPass(),
      fx_passes.CastInputsBf16ToF32Pass(),
  ]

  # Debuginfo is not injected automatically by odml_torch. Only inject
  # debuginfo via fx pass when using torch_xla.
  if ai_edge_torch.config.use_torch_xla:
    passes += [
        fx_passes.InjectMlirDebuginfoPass(),
        fx_passes.CanonicalizePass(),
    ]

  exported_program = fx_infra.run_passes(exported_program, passes)
  return exported_program


def _warn_training_modules(signatures: list[signature.Signature]):
  """Warns the user if the module is in training mode (.eval not called)."""
  for sig in signatures:
    if not sig.module.training:
      continue

    message = (
        "Your model {sig_name}is converted in training mode. Please set the"
        " module in evaluation mode with `module.eval()` for better on-device"
        " performance and compatibility."
    )
    if len(signatures) == 1 and sig.name == model.DEFAULT_SIGNATURE_NAME:
      # User does not specify any signature names explicitly.
      message = message.format(sig_name="")
    else:
      message = message.format(sig_name=f'"{sig.name}" ')

    logging.warning(message)


def convert_signatures(
    signatures: list[signature.Signature],
    *,
    strict_export: Union[Literal["auto"], bool] = True,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: Optional[dict[str, Any]] = None,
    _saved_model_dir: Optional[str] = None,
) -> model.TfLiteModel:
  """Converts a list of `signature.Signature`s and embeds them into one `model.TfLiteModel`.

  Args:
      signatures: The list of 'signature.Signature' objects containing PyTorch
        modules to be converted.
      strict_export: Experimental `strict` arg for torch.export.export. When
        enabled, the export function will trace the program through TorchDynamo
        and ensure the soundness of the exported graph. When
        strict_export="auto", the function will try to export module in both
        modes and use the first one succeeds for downstream conversion.
      quant_config: User-defined quantization method and scheme of the model.
      _tfl_converter_flags: A nested dictionary allowing setting flags for the
        underlying tflite converter.
      _saved_model_dir: Directory for the intermediate saved model. If not
        specified, a random temporary directory would be used.

  Returns:
    The converted `model.TfLiteModel` object.
  """
  if _tfl_converter_flags is None:
    _tfl_converter_flags = {}

  _warn_training_modules(signatures)

  def export(**kwargs):
    nonlocal strict_export
    if strict_export == "auto":
      try:
        exported_program = torch.export.export(**kwargs, strict=True)
      except Exception:
        logging.warning(
            "torch.export.export(..., strict=True) failed. Retrying with"
            " strict=False"
        )
        exported_program = torch.export.export(**kwargs, strict=False)
    elif not strict_export:
      exported_program = torch.export.export(**kwargs, strict=False)
    else:
      exported_program = torch.export.export(**kwargs, strict=True)

    exported_program = fx_infra.graph_utils.reset_from_node_meta(
        exported_program
    )

    exported_program = fx_infra.safe_run_decompositions(
        exported_program,
        fx_infra.decomp.pre_convert_decomp(),
    )
    return exported_program

  exported_programs: torch.export.ExportedProgram = [
      export(
          mod=sig.module,
          args=sig.args,
          kwargs=sig.kwargs,
          dynamic_shapes=sig.dynamic_shapes,
      )
      for sig in signatures
  ]

  # Apply default fx passes
  exported_programs = list(map(_run_convert_passes, exported_programs))
  tflite_model = lowertools.exported_programs_to_tflite(
      exported_programs,
      signatures,
      quant_config=quant_config,
      _tfl_converter_flags=_tfl_converter_flags,
      _saved_model_dir=_saved_model_dir,
  )

  return model.TfLiteModel(tflite_model)


def aot_compile(
    compilation_configs: list[litert_types.CompilationConfig],
    cpu_model: model.TfLiteModel,
) -> litert_types.CompilationResult:
  """Compiles the given CPU model.

  Args:
    compilation_configs: The list of compilation configs to use.
    cpu_model: The CPU model to compile.

  Returns:
    The compilation result.
  """
  litert_model = litert_types.Model.create_from_bytes(cpu_model.tflite_model())
  return aot_compile_lib.aot_compile(
      litert_model,
      config=compilation_configs,
  )
