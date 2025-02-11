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
import torch
import re


def _class_fullname(cls):
  module = cls.__module__
  if module == "builtins":
    return cls.__qualname__
  return module + "." + cls.__qualname__


def _get_hierarchy(node: torch.fx.Node):
  nn_module_stack = node.meta.get("nn_module_stack", {})
  layers = []
  for name, layer in nn_module_stack.values():
    iid = ("_" + name.split(".")[-1]) if name else ""
    layer_str = layer if isinstance(layer, str) else _class_fullname(layer)
    layers.append(layer_str + iid)

  hierachy_str = "/".join(layers) + ";"
  return hierachy_str


def _get_canonical_filename(filename):
  """Remove unnecessary path prefix to make the filename more readable.

  This should be factored out so that pattern is a global option that a user
  can override.

  Args:
    filename: The filename to canonicalize.

  Returns:
    The canonicalized filename.
  """

  # TODO(yijieyang): We should add a config option to provide a regex to strip
  # from the debug info. Currently absolute path is used.
  return filename


def build_mlir_file_debuginfo(node: torch.fx.Node):
  """Build the file and line info for the given node's lowerings in MLIR."""

  if not node.stack_trace:
    return None, None

  # Note: This uses internal APIs and may break in the future.
  pt_trace = torch.fx.graph._parse_stack_trace(node.stack_trace)
  if pt_trace is None:
    return None, None
  return _get_canonical_filename(pt_trace.file), int(pt_trace.lineno)


def build_nodename_debuginfo(node: torch.fx.Node):
  """Build the fx node name for the given node's lowerings in MLIR."""
  history = node.meta.get("from_node", [])
  if not history:
    return None
  if len(history) > 1:
    return history[1][0]
  if hasattr(history[0], "name"):  # torch 2.6.0+
    return history[0].name
  return None


def build_mlir_debuginfo(node: torch.fx.Node):
  """Build the debuginfo string for the given node's lowerings in MLIR."""

  if not hasattr(node, "meta") or "nn_module_stack" not in node.meta:
    return None

  return _get_hierarchy(node)
