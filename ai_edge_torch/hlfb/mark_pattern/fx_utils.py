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
"""FX graph utilities for pattern matching clean ups."""

from ai_edge_torch import fx_infra
import torch

remove_dangling_args = fx_infra.graph_utils.remove_dangling_args
remove_assert_tensor_metadata_nodes = (
    fx_infra.graph_utils.remove_assert_tensor_metadata_nodes
)

def is_clone_op(node: torch.fx.Node) -> bool:
  """Checks if the node is a clone op."""
  return (
      node.op == "call_function" and node.target == torch.ops.aten.clone.default
  )


def remove_clone_ops(gm: torch.fx.GraphModule):
  """Removes clone ops from the graph.

  torch export adds additional aten.clone nodes to produce contiguous in memory
  tensors depending on tensor sizes for runtime efficiency. However, these
  unpredictable clone nodes can break the pattern matching. Thus remove all
  clones in model and pattern graphs.

  Args:
    gm: The graph module to remove clone ops from.

  Returns:
    The graph module with clone ops removed.
  """
  for node in gm.graph.nodes:
    if is_clone_op(node):
      node.replace_all_uses_with(node.args[0])
      gm.graph.erase_node(node)

  gm.graph.lint()
  gm.recompile()
  return gm
