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

from ai_edge_torch._convert import signature as signature_module
import tensorflow as tf
import torch


def _torch_to_tf_variable(torch_tensor: torch.Tensor):
  if not torch_tensor.is_contiguous():
    torch_tensor = torch_tensor.contiguous()

  try:
    dlpack_capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
    tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack_capsule)
  except Exception:
    logging.info(
        "Can not use dlpack to convert torch tensors. Falling back to numpy."
    )
    nparray = torch_tensor.cpu().detach().numpy()
    tf_tensor = tf.convert_to_tensor(nparray)

  return tf.Variable(tf_tensor, trainable=False)


def _get_states(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature_module.Signature],
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
          or input_spec.kind
          == torch.export.graph_signature.InputKind.USER_INPUT
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


def gather_state_dict(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature_module.Signature],
):
  deduped_tensor_map = {}

  for _, tensor, _ in _get_states(exported_programs, signatures):
    unique_id = _tensor_unique_id(tensor)
    deduped_tensor_map[unique_id] = _torch_to_tf_variable(tensor)

  state_dict = {}
  for signature, tensor, input_spec in _get_states(
      exported_programs, signatures
  ):
    unique_id = _tensor_unique_id(tensor)
    state_dict[signature.name + "_" + input_spec.target] = deduped_tensor_map[
        unique_id
    ]

  return state_dict, list(deduped_tensor_map.values())
