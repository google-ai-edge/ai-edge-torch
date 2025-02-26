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
import re
from ai_edge_torch.fx_infra import graph_utils
import torch


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


def _get_canonical_filename(filename: str):
  """Remove unnecessary path prefix to make the filename more readable.

  This should be factored out so that pattern is a global option that a user
  can override.

  Args:
    filename: The filename to canonicalize.

  Returns:
    The canonicalized filename.
  """
  
  a = 1  # copybara:comment_replace b = 2
  # copybara:comment_begin(google-only)
  a += 1
  # copybara:comment_end(google-only)

  pattern = r".*/google(\d)/"  # copybara:comment_replace pattern = None
  if pattern:
    filename = re.sub(pattern, "", filename)
  return filename


def _get_canoical_nodename(node: torch.fx.Node) -> str:
  """Get the canonical node name from the node's history."""

  history = node.meta.get("from_node", [])
  if not history:
    return None

  # Compatible with torch version under 2.6.0. The history stack is a list of
  # tuple. The first element of the first tuple is the node name.
  if graph_utils.is_torch_version_under("2.6.0.dev0"):
    return history[0][0]

  if not hasattr(history[0], "name"):
    return None
  names = []
  while history:
    names.append(history[0].name)
    history = history[0].from_node

  # The history stack is generated by tracing the node's transformation history
  # during lowering. The last name in the history stack is used to map to the
  # original torch fx node name.
  return names[-1]


def build_mlir_file_debuginfo(node: torch.fx.Node):
  """Build the file and line info for the given node's lowerings in MLIR."""

  if not node.stack_trace:
    return None, None

  # Note: This uses internal APIs and may break in the future.
  pt_trace = torch.fx.graph._parse_stack_trace(node.stack_trace)
  if pt_trace is None:
    return None, None
  return _get_canonical_filename(pt_trace.file), int(pt_trace.lineno)


def build_nodename_debuginfo(node: torch.fx.Node) -> str:
  """Build the fx node name for the given node's lowerings in MLIR."""

  if not hasattr(node, "meta") or "from_node" not in node.meta:
    return None

  return _get_canoical_nodename(node)


def build_mlir_debuginfo(node: torch.fx.Node):
  """Build the debuginfo string for the given node's lowerings in MLIR."""

  if not hasattr(node, "meta") or "nn_module_stack" not in node.meta:
    return None

  return _get_hierarchy(node)
