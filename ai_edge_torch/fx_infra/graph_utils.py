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
import torch


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
