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

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree


@dataclasses.dataclass
class Signature:
  name: str
  module: torch.nn.Module
  sample_args: tuple[torch.Tensor]
  sample_kwargs: dict[str, torch.Tensor]
  dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None

  @property
  def _normalized_sample_args_kwargs(self):
    args, kwargs = self.sample_args, self.sample_kwargs
    if args is not None:
      if not isinstance(args, tuple):
        # TODO(b/352584188): Check value types
        raise ValueError("sample_args must be a tuple of torch tensors.")
    if kwargs is not None:
      if not isinstance(kwargs, dict) or not all(
          isinstance(key, str) for key in kwargs.keys()
      ):
        # TODO(b/352584188): Check value types
        raise ValueError("sample_kwargs must be a dict of string to tensor.")
    args = args if args is not None else tuple()
    kwargs = kwargs if kwargs is not None else {}
    return args, kwargs

  @property
  def flat_arg_names(self) -> list[str]:
    spec = pytree.tree_flatten(self._normalized_sample_args_kwargs)[1]
    args_spec, kwargs_spec = spec.children_specs
    names = []
    for i in range(args_spec.num_leaves):
      names.append(f"args_{i}")

    kwargs_names = self._flat_kwarg_names(
        kwargs_spec.children_specs, kwargs_spec.context
    )
    names.extend(kwargs_names)
    return names

  def _flat_kwarg_names(self, specs, context) -> List[str]:
    flat_names = []
    if context is None:
      for i, spec in enumerate(specs):
        if spec.children_specs:
          flat_names.extend([
              f"{i}_{name}"
              for name in self._flat_kwarg_names(
                  spec.children_specs, spec.context
              )
          ])
        else:
          flat_names.append(f"{i}")
    else:
      flat_ctx = self._flatten_list(context)
      for prefix, spec in zip(flat_ctx, specs):
        leaf_flat_names = self._flat_kwarg_names(
            spec.children_specs, spec.context
        )
        if leaf_flat_names:
          flat_names.extend([f"{prefix}_{name}" for name in leaf_flat_names])
        else:
          flat_names.append(prefix)

    return flat_names

  def _flatten_list(self, l: List) -> List:
    flattened = []
    for item in l:
      if isinstance(item, list):
        flattened.extend(self._flatten_list(item))
      else:
        flattened.append(item)
    return flattened

  @property
  def flat_args(self) -> tuple[Any]:
    args, kwargs = self._normalized_sample_args_kwargs
    return tuple([*args, *kwargs.values()])
