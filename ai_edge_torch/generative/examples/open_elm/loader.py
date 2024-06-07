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
# Data loading utility for OpenELM models.
from dataclasses import dataclass
import glob
import os
from typing import Callable, Dict

from safetensors import safe_open
import torch

from ai_edge_torch.generative.layers import model_config


def load_safetensors(full_path: str):
  """Loads safetensors into a single state dictionary.

  Args:
    full_path (string): the directory that contains the safetensor files.

  Returns:
    A state dictionary contating loaded tensors.

  Raises:
    ValueError: If no tensors are loaded from the provided directory or file.
  """
  pattern = (
      os.path.join(full_path, "*.safetensors")
      if os.path.isdir(full_path)
      else full_path
  )
  files = []
  for file in glob.glob(pattern):
    files.append(file)

  tensors = {}
  for file in files:
    with safe_open(file, framework="pt") as fp:
      for k in fp.keys():
        assert k not in tensors
        tensors[k] = fp.get_tensor(k)

  if not tensors:
    raise ValueError("Failed to load SafeTensors.")
  return tensors


def load_pytorch_statedict(full_path: str):
  """Loads state dictionary binaries into a single state dictionary.

  Args:
    full_path (string): the directory that contains the bin files.

  Returns:
    A state dictionary contating loaded tensors.

  Raises:
    ValueError: If no tensors are loaded from the provided directory or file.
  """
  pattern = os.path.join(full_path, "*.bin") if os.path.isdir(full_path) else full_path
  files = []
  for file in glob.glob(pattern):
    files.append(file)

  tensors = {}
  for file in files:
    this_file_tensors = torch.load(file)
    for k in this_file_tensors:
      assert k not in tensors
    tensors.update(this_file_tensors)

  if not tensors:
    raise ValueError("Failed to load torch bin files.")
  return tensors


class ModelLoader:
  """A utility class for loading and converting model checkpoints to the
  Edge Generative API layer format.
  """

  @dataclass
  class TensorNames:
    embedding: str

    pre_attn_norm: str
    attn_qkv_proj: str
    attn_query_norm: str
    attn_key_norm: str
    attn_output_proj: str

    pre_ff_norm: str
    ff_gate_up_proj: str
    ff_down_proj: str

    final_norm: str

  def __init__(self, file_name: str, names: TensorNames) -> None:
    """ModelLoader constructor. Can be used to load multiple models of the same
    type.

    Args:
        file_name (str): Path to the checkpoint. Can be a directory or an
          exact file.
        names (TensorNames): An instance of `TensorNames` to determine mappings.
    """
    self._file_name = file_name
    self._names = names
    self._loader = self._get_loader()

  def load(self, model: torch.nn.Module, strict: bool = True):
    """Load the model from the checkpoint

    Args:
        model (torch.nn.Module): The pytorch model that needs to be loaded.
        strict (bool, optional): Whether the converted keys are strictly
          matched. Defaults to True.

    Raises:
        ValueError: If conversion results in unmapped tensors and strict mode is
          enabled.
    """
    state = self._loader(self._file_name)
    converted_state = dict()
    if self._names.embedding is not None:
      converted_state["tok_embedding.weight"] = state.pop(
          f"{self._names.embedding}.weight"
      )
    if self._names.final_norm is not None:
      final_norm_name = self._names.final_norm
      converted_state["final_norm.weight"] = state.pop(f"{final_norm_name}.weight")

    for i in range(model.config.num_layers):
      self._map_norm(i, model.config, state, converted_state)
      self._map_feedforward(i, model.config, state, converted_state)
      self._map_attention(i, model.config, state, converted_state)

    if strict and state:
      raise ValueError(
          f"Failed to map all tensor. Remaing tensor are: {list(state.keys())}"
      )
    model.load_state_dict(converted_state, strict=strict)

  def _get_loader(self) -> Callable[[str], Dict[str, torch.Tensor]]:
    """A best effort method for finding appropriate state loader.

    Raises:
        ValueError: If it fails to find an appropriate loader.

    Returns:
        Callable[[str], Dict[str, torch.Tensor]]: State loader to be used.
    """
    if os.path.isdir(self._file_name):
      if glob.glob(os.path.join(self._file_name, "*.safetensors")):
        return load_safetensors
      if glob.glob(os.path.join(self._file_name, "*.bin")):
        return load_pytorch_statedict

    if self._file_name.endswith(".safetensors"):
      return load_safetensors

    if self._file_name.endswith(".bin"):
      return load_pytorch_statedict

    raise ValueError(f"File format not supported.")

  def _map_feedforward(
      self,
      idx: int,
      config: model_config.ModelConfig,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    ff_gate_up_proj_name = self._names.ff_gate_up_proj.format(idx)
    ff_down_proj_name = self._names.ff_down_proj.format(idx)
    up_gate = state.pop(f"{ff_gate_up_proj_name}.weight")
    gate, up = torch.chunk(up_gate, 2)
    converted_state[f"{prefix}.ff.w3.weight"] = up
    converted_state[f"{prefix}.ff.w2.weight"] = state.pop(f"{ff_down_proj_name}.weight")
    converted_state[f"{prefix}.ff.w1.weight"] = gate

  def _map_attention(
      self,
      idx: int,
      config: model_config.ModelConfig,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    qkv_name = self._names.attn_qkv_proj.format(idx)
    converted_state[f"{prefix}.atten_func.qkv_projection.weight"] = state.pop(
        f"{qkv_name}.weight"
    )
    o_name = self._names.attn_output_proj.format(idx)
    converted_state[f"{prefix}.atten_func.output_projection.weight"] = state.pop(
        f"{o_name}.weight"
    )

  def _map_norm(
      self,
      idx: int,
      config: model_config.ModelConfig,
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    pre_attn_norm_name = self._names.pre_attn_norm.format(idx)
    converted_state[f"{prefix}.pre_atten_norm.weight"] = state.pop(
        f"{pre_attn_norm_name}.weight"
    )
    pre_ff_norm_name = self._names.pre_ff_norm.format(idx)
    converted_state[f"{prefix}.pre_ff_norm.weight"] = state.pop(
        f"{pre_ff_norm_name}.weight"
    )
    query_norm_name = self._names.attn_query_norm.format(idx)
    converted_state[f"{prefix}.atten_func.query_norm.weight"] = state.pop(
        f"{query_norm_name}.weight"
    )
    key_norm_name = self._names.attn_key_norm.format(idx)
    converted_state[f"{prefix}.atten_func.key_norm.weight"] = state.pop(
        f"{key_norm_name}.weight"
    )
