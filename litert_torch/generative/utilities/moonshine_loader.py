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
# Common utility functions for data loading etc.
from dataclasses import dataclass
import glob
import os
from typing import Callable, Dict

import h5py
import torch


def transpose_if_needed(t):
  """We assume the file is from Keras, i.e. channel last format."""
  if len(t.shape) > 2:
    return t.permute(2, 1, 0)
  return t


def load_h5_statedict(full_path: str):
  """Loads the HDF5 DataSets into a single dctionary.

  Args:
    full_path (string): the HDF5 filename or directory that contains the HDF5
      files.

  Returns:
    A state dictionary contating loaded tensors.

  Raises:
    ValueError: If no tensors are loaded from the provided directory or file.
  """
  pattern = (
      os.path.join(full_path, "*.h5") if os.path.isdir(full_path) else full_path
  )
  files = []
  for file in glob.glob(pattern):
    files.append(file)

  tensors = {}

  def collect_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
      tensors[name] = transpose_if_needed(torch.from_numpy(obj[:]))

  for file in files:
    with h5py.File(file) as f:
      f.visititems(collect_datasets)

  if not tensors:
    raise ValueError("Failed to load HDF5 file.")
  return tensors


class ModelLoader:
  """Utility class for loading and converting checkpoints to ODML transformer layer format."""

  @dataclass
  class TensorNames:
    conv1D_0: str = None
    conv1D_1: str = None
    conv1D_2: str = None
    group_norm: str = None

  def __init__(self, file_name: str, names: TensorNames) -> None:
    """ModelLoader constructor.

    Can be used to load multiple models of the same type.

    Args:
        file_name (str): Path to the checkpoint. Can be a directory or an exact
          file.
        names (TensorNames): An instance of `TensorNames` to determine mappings.
    """
    self._file_name = file_name
    self._names = names
    self._loader = load_h5_statedict

  def load(
      self,
      model: torch.nn.Module,
      strict: bool = True,
  ):
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

    if isinstance(self._names, ModelLoader.TensorNames):
      converted_state = self._do_load(model, state, self._names)
    else:
      raise ValueError(f"Unkown type for names: {type(self._names)}")

    if strict and state:
      raise ValueError(
          "Failed to map all tensor. Remaining tensor are:"
          f" {list(state.keys())}"
      )
    model.load_state_dict(converted_state, strict=strict)

  def _do_load(self, model, state, names, additional_prefix=""):
    """Load the model from the checkpoint

    Args:
        model (torch.nn.Module): The pytorch model that needs to be loaded.
        state (Dict[str, torch.Tensor]): The pytorch state dictionary
        names (TensorNames]): The TensorNames for the model we are loading.

    Returns:
        Dict[str, torch.Tensor]: Map of name to tensor for loading.
    """
    converted_state = dict()
    if names.conv1D_0 is not None:
      converted_state["conv1.weight"] = state.pop(f"{names.conv1D_0}/0")
      if f"{names.conv1D_0}/1" in state:
        converted_state["conv1.bias"] = state.pop(f"{names.conv1D_0}/1")

    if names.conv1D_1 is not None:
      converted_state["conv2.weight"] = state.pop(f"{names.conv1D_1}/0")
      if f"{names.conv1D_1}/1" in state:
        converted_state["conv2.bias"] = state.pop(f"{names.conv1D_1}/1")

    if names.conv1D_2 is not None:
      converted_state["conv3.weight"] = state.pop(f"{names.conv1D_2}/0")
      if f"{names.conv1D_2}/1" in state:
        converted_state["conv3.bias"] = state.pop(f"{names.conv1D_2}/1")

    if names.group_norm is not None:
      group_norm_name = names.group_norm
      converted_state[f"group_norm.weight"] = state.pop(f"{group_norm_name}/0")
      if f"{group_norm_name}/1" in state:
        converted_state["group_norm.bias"] = state.pop(f"{group_norm_name}/1")

    return converted_state
