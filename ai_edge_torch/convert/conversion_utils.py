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

import copy
from dataclasses import dataclass
import gc
import itertools
import logging
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch_xla import stablehlo

from ai_edge_torch.generative.quantize.ai_edge_quantizer_glue import translate_recipe  # NOQA
from ai_edge_torch.quantize import quant_config as qcfg

try:
  import tensorflow as tf
  from tensorflow.compiler.tf2xla.python import xla as tfxla

  from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb  # isort:skip
except ImportError:
  logging.error(
      "This module needs tensorflow with xla support.\n"
      "Please install tensorflow with `pip install tf-nightly`.\n"
  )
  raise

DEFAULT_SIGNATURE_NAME = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY


@dataclass
class Signature:
  name: str
  module: torch.nn.Module
  sample_args: tuple[torch.Tensor]
  dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None


def exported_program_to_stablehlo_bundle(
    exported_program: torch.export.ExportedProgram, sample_args: tuple[torch.Tensor]
) -> stablehlo.StableHLOModelBundle:
  # Setting export_weights to False here so that pytorch/xla avoids copying the weights
  # to a numpy array which would lead to memory bloat. This means that the state_dict
  # in the returned bundle is going to be empty.
  return stablehlo.exported_program_to_stablehlo(
      exported_program,
      stablehlo.StableHLOExportOptions(
          override_tracing_arguments=sample_args, export_weights=False
      ),
  )._bundle


def _torch_to_tf_tensor(torch_tensor: torch.Tensor):
  if not torch_tensor.is_contiguous():
    torch_tensor = torch_tensor.contiguous()

  try:
    dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
    tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack_capsule)
  except Exception:
    logging.info("Can not use dlpack to convert torch tensors. Falling back to numpy.")
    nparray = torch_tensor.cpu().detach().numpy()
    tf_tensor = tf.convert_to_tensor(nparray)

  return tf_tensor


def _get_states(
    exported_programs: list[torch.export.ExportedProgram], signatures: list[Signature]
):
  for exported_program, signature in zip(exported_programs, signatures):
    args, _ = exported_program.example_inputs
    # Calling this to get **all** the state including model buffers.
    _flat_input_args = exported_program._graph_module_flat_inputs(args, {})
    for tensor, input_spec in zip(
        _flat_input_args, exported_program.graph_signature.input_specs
    ):
      # Only interested in Tensors that are part of the state (and not user input).
      if (
          not isinstance(tensor, torch.Tensor)
          or input_spec.kind == torch.export.graph_signature.InputKind.USER_INPUT
      ):
        continue
      yield signature, tensor, input_spec


def _tensor_unique_id(tensor: torch.Tensor):
  return (
      str(tensor.device),
      tensor.shape,
      tensor.stride(),
      tensor.untyped_storage().data_ptr(),
  )


def _gather_state_dict(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[Signature],
):
  deduped_tensor_map = {}

  for _, tensor, _ in _get_states(exported_programs, signatures):
    unique_id = _tensor_unique_id(tensor)
    deduped_tensor_map[unique_id] = _torch_to_tf_tensor(tensor)

  state_dict = {}
  for signature, tensor, input_spec in _get_states(exported_programs, signatures):
    unique_id = _tensor_unique_id(tensor)
    state_dict[signature.name + "_" + input_spec.target] = deduped_tensor_map[unique_id]

  return state_dict


def merge_stablehlo_bundles(
    bundles: list[stablehlo.StableHLOModelBundle],
    signatures: list[Signature],
    exported_programs: list[torch.export.ExportedProgram],
) -> stablehlo.StableHLOGraphModule:
  state_dict = _gather_state_dict(exported_programs, signatures)

  new_bundle = stablehlo.StableHLOModelBundle(
      state_dict=state_dict, additional_constants=[], stablehlo_funcs=[]
  )

  for bundle, signature in zip(bundles, signatures):
    const_offset = len(new_bundle.additional_constants)
    for func in bundle.stablehlo_funcs:
      func.meta.name = signature.name + "_" + func.meta.name
      for loc in func.meta.input_locations:
        if loc.type_ == stablehlo.VariableType.CONSTANT:
          loc.position += const_offset
        elif loc.type_ == stablehlo.VariableType.PARAMETER:
          loc.name = signature.name + "_" + loc.name
      new_bundle.stablehlo_funcs.append(func)
    new_bundle.additional_constants.extend(bundle.additional_constants)
  return stablehlo.StableHLOGraphModule(new_bundle)


def _get_shape_with_dynamic(signature: stablehlo.VariableSignature):
  shape = copy.copy(signature.shape)
  for i in signature.dynamic_dims:
    shape[i] = None
  return shape


def _wrap_as_tf_func(
    func: stablehlo.StableHLOFunc, bundle: stablehlo.StableHLOModelBundle
):
  def inner(*args):
    type_info = [sig.dtype for sig in func.meta.output_signature]
    shape_info = [_get_shape_with_dynamic(sig) for sig in func.meta.output_signature]
    call_args = stablehlo._extract_call_parameters(args, func.meta, bundle)
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=type_info,
        Sout=shape_info,
        function_list=[],
        module=func.bytecode,
    )

  return inner


def _make_tf_function(
    shlo_graph_module: stablehlo.StableHLOGraphModule,
    bundle: stablehlo.StableHLOModelBundle = None,
):
  bundle = shlo_graph_module._bundle if bundle is None else bundle
  return [
      _wrap_as_tf_func(func, bundle)
      for func in shlo_graph_module._bundle.stablehlo_funcs
  ]


def _make_tf_signature(
    meta: stablehlo.StableHLOFunctionMeta,
) -> list[tf.TensorSpec]:
  input_pos_to_spec = {
      loc.position: spec
      for loc, spec in itertools.chain(
          zip(meta.input_locations, meta.input_signature), meta.unused_inputs
      )
      if loc.type_ == stablehlo.VariableType.INPUT_ARG
  }
  primitive_type_to_tf_type = {"int": "int32", "float": "float32"}
  ret: list[tf.TensorSpec] = []
  for i in range(len(input_pos_to_spec)):
    spec = input_pos_to_spec[i]
    shape = _get_shape_with_dynamic(spec)
    ret.append(
        tf.TensorSpec(
            shape=shape,
            dtype=primitive_type_to_tf_type[spec.dtype]
            if spec.dtype in primitive_type_to_tf_type
            else spec.dtype,
            name=f"args_{i}",
        )
    )
  return ret


def _apply_tfl_backdoor_flags(
    converter: tf.lite.TFLiteConverter, tfl_converter_flags: dict
):
  def _set_converter_flag(path: list):
    if len(path) < 2:
      raise ValueError("Expecting at least two values in the path.")

    target_obj = converter
    for idx in range(len(path) - 2):
      target_obj = getattr(target_obj, path[idx])

    setattr(target_obj, path[-2], path[-1])

  def _iterate_dict_tree(flags_dict: dict, path: list):
    for key, value in flags_dict.items():
      path.append(key)
      if isinstance(value, dict):
        _iterate_dict_tree(value, path)
      else:
        path.append(value)
        _set_converter_flag(path)
        path.pop()
      path.pop()

  _iterate_dict_tree(tfl_converter_flags, [])


def _set_tfl_converter_quant_flags(
    converter: tf.lite.TFLiteConverter, quant_config: qcfg.QuantConfig
):
  if quant_config is not None:
    quantizer_mode = quant_config._quantizer_mode
    if quantizer_mode == qcfg.QuantConfig._QuantizerMode.PT2E_DYNAMIC:
      converter._experimental_qdq_conversion_mode = "DYNAMIC"
    elif quantizer_mode == qcfg.QuantConfig._QuantizerMode.PT2E_STATIC:
      converter._experimental_qdq_conversion_mode = "STATIC"


def convert_stablehlo_to_tflite(
    shlo_graph_module: stablehlo.StableHLOGraphModule,
    signatures: list[Signature],
    *,
    quant_config: Optional[qcfg.QuantConfig] = None,
    _tfl_converter_flags: dict = {},
) -> None:
  """Converts a StableHLOGraphModule to a tflite model.
  Args:
    shlo_graph_module - model to export and save
    signatures: List of signatures from which names of the signatures is extracted.
    quant_config: User-defined quantization method and scheme of the model.
    _tfl_converter_flags: A nested dictionary allowing setting flags for the underlying tflite converter.
  """

  bundle = shlo_graph_module._bundle
  tf_module = tf.Module()
  bundle.state_dict = {
      k: tf.Variable(v, trainable=False) for k, v in bundle.state_dict.items()
  }
  bundle.additional_constants = [
      tf.Variable(v, trainable=False) for v in bundle.additional_constants
  ]
  tf_signatures: list[list[tf.TensorSpec]] = list(
      _make_tf_signature(func.meta) for func in bundle.stablehlo_funcs
  )

  tf_functions = _make_tf_function(shlo_graph_module, bundle)

  tf_module.f = []
  for tf_sig, func in zip(tf_signatures, tf_functions):
    tf_module.f.append(
        tf.function(
            func,
            input_signature=tf_sig,
        )
    )

  tf_module._variables = list(bundle.state_dict.values()) + bundle.additional_constants
  del bundle
  gc.collect()

  tf_concrete_funcs = [
      func.get_concrete_function(*tf_sig)
      for func, tf_sig in zip(tf_module.f, tf_signatures)
  ]

  # We need to temporarily save since TFLite's from_concrete_functions does not
  # allow providing names for each of the concrete functions.
  with tempfile.TemporaryDirectory() as temp_dir_path:
    tf.saved_model.save(
        tf_module,
        temp_dir_path,
        signatures={
            sig.name: tf_concrete_funcs[idx] for idx, sig in enumerate(signatures)
        },
    )
    # Clean up intermediate memory early.
    del tf_module
    del tf_concrete_funcs
    gc.collect()

    converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir_path)
    converter._set_original_model_type(conversion_metadata_fb.ModelType.PYTORCH)
    converter._experimental_enable_composite_direct_lowering = True

    _set_tfl_converter_quant_flags(converter, quant_config)
    if (
        quant_config is not None
        and quant_config._quantizer_mode
        == quant_config._QuantizerMode.AI_EDGE_QUANTIZER
    ):
      translated_recipe = translate_recipe.translate_to_ai_edge_recipe(
          quant_config.generative_recipe
      )

    _apply_tfl_backdoor_flags(converter, _tfl_converter_flags)

    tflite_model = converter.convert()

    if (
        quant_config is not None
        and quant_config._quantizer_mode
        == quant_config._QuantizerMode.AI_EDGE_QUANTIZER
    ):
      tflite_model = translate_recipe.quantize_model(tflite_model, translated_recipe)

  return tflite_model
