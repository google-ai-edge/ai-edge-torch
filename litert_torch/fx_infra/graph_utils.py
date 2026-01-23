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
"""FX graph utilities."""

from packaging import version
import torch
from torch.fx import traceback


def remove_dangling_args(graph_module: torch.fx.GraphModule):
  """Removes dangling args from the graph."""
  for node in graph_module.graph.nodes:
    if node.op == "placeholder" and not node.users:
      graph_module.graph.erase_node(node)

  graph_module.graph.lint()
  graph_module.recompile()
  return graph_module


def remove_assert_tensor_metadata_nodes(graph_module: torch.fx.GraphModule):
  """Removes aten._assert_tensor_metadata nodes from the graph.

  This op is inserted by torch.export to check tensor metadata on custom ops. It
  can break patten matching and lowering.
  """
  for node in graph_module.graph.nodes:
    if node.target == torch.ops.aten._assert_tensor_metadata.default:
      graph_module.graph.erase_node(node)

  graph_module.graph.lint()
  graph_module.recompile()
  return graph_module


def is_torch_version_under(torch_version: str) -> bool:
  """Checks if the current torch version is under the given version."""
  if not torch_version:
    raise ValueError("torch_version cannot be empty.")
  current_version = version.parse(torch.__version__)
  compared_version = version.parse(torch_version)
  return current_version < compared_version


def reset_from_node_meta(ep: torch.export.ExportedProgram):
  """Resets the "from_node" meta field to fx node name only for the exported program."""

  for node in ep.graph.nodes:
    if not hasattr(node, "meta") or "from_node" not in node.meta:
      continue
    if is_torch_version_under("2.6.0.dev0"):
      # For torch version under 2.6.0, the history stack is a list of tuple. We
      # will only keep the current node's name in the history stack.
      history = [(node.name,)]
    else:
      # Clean up the history stack by keeping only the current node info (fx
      # node name and graph id) in a list of size 1. Clear the "from_node" field
      # to prevent redundant additions to the history stack.
      history = [traceback.NodeSource(node)]
      history[0].from_node = []
    node.meta["from_node"] = history
  return ep
