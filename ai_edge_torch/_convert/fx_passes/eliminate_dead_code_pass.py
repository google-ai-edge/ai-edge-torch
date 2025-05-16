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
"""Pass to eliminate dead code for ai-edge-torch conversion."""


from ai_edge_torch import fx_infra
import torch


class EliminateDeadCodePass(fx_infra.PassBase):
  """Eliminates dead code with dedicated rules for ai-edge-torch conversion."""

  def call(self, graph_module: torch.fx.GraphModule):
    def is_impure_node(node: torch.fx.Node):
      # Starting from torch 2.7.0, random torch ops with
      # _nondeterministic_seeded set are no longer considered pure. However,
      # for conversion, unused random ops/tensors should still be removed.
      if getattr(node.target, "_nondeterministic_seeded", False):
        return False
      return node.is_impure()

    try:
      graph_module.graph.eliminate_dead_code(is_impure_node)
    except TypeError:
      # eliminate_dead_code has no is_impure_node input in old torch versions.
      pass

    return fx_infra.PassResult(graph_module, True)
