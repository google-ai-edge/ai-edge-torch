# Copyright 2025 The AI Edge Torch Authors.
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
"""Torch export utilities for testing."""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils import _pytree as pytree


def export_with_tensor_inputs_only(
    model: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.export.ExportedProgram:
  """Exports a PyTorch model, treating only tensor inputs as export inputs.

  This function takes a PyTorch model and its input arguments (positional and
  keyword) and exports it using `torch.export.export`. However, it modifies
  the export process such that only the `torch.Tensor` arguments in the
  inputs are considered as export inputs to the exported graph. All other
  argument types (e.g., scalars, lists, tuples containing non-tensors) are
  treated as constants.

  This is useful for testing scenarios where you want to export a model but
  want to avoid issues that might arise from non-tensor inputs
  being treated as variables, or when you specifically want to focus on the
  graph structure based on tensor operations.

  Args:
    model: The PyTorch `nn.Module` to be exported.
    args: A tuple of positional arguments to be passed to the model's `forward`
      method.
    kwargs: A dictionary of keyword arguments to be passed to the model's
      `forward` method.

  Returns:
    torch.export.ExportedProgram: The exported program representing the model
    computation with only tensor inputs being export inputs.
  """
  flatten_args, treespec = pytree.tree_flatten([args, kwargs])

  export_args = []
  indices = []
  for i, arg in enumerate(flatten_args):
    if isinstance(arg, torch.Tensor):
      export_args.append(arg)
      indices.append(i)

  class ModuleWrapper(torch.nn.Module):

    def __init__(self, func, original_args, original_kwargs):
      super().__init__()
      self.original_args = list(flatten_args)
      self.func = func

    def forward(self, *export_args):
      flatten_args = self.original_args.copy()
      for i, arg in zip(indices, export_args):
        flatten_args[i] = arg
      args, kwargs = pytree.tree_unflatten(flatten_args, treespec)
      return self.func(*args, **kwargs)

  export_args = tuple(export_args)
  export_kwargs = {}
  return torch.export.export(
      ModuleWrapper(model, args, kwargs).eval(),
      export_args,
      export_kwargs,
  )
