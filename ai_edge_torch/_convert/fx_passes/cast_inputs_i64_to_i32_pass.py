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
"""Pass to cast all inputs with torch.int64 type to torch.int32."""


from ai_edge_torch import fx_infra
import torch


def cast_i32(x):
  # return x.to(torch.int32)
  return x.to(torch.float32)


class CastInputsI64ToI32Pass(fx_infra.ExportedProgramPassBase):
  """This pass casts all inputs with torch.int64 type to torch.int32."""

  def call(self, exported_program: torch.export.ExportedProgram):
    modified = False
    for node in exported_program.graph.nodes:
      if (
          node.op in ("placeholder", "call_function")
          and node.meta.get("val") is not None
          and node.meta.get("val").dtype == torch.int64
      ):
        if not node.users:
          continue

        modified = True
        user = next(iter(node.users))
        with exported_program.graph.inserting_before(user):
          cast_node = exported_program.graph.call_function(
              cast_i32,
              (node,),
          )
          node.replace_all_uses_with(cast_node)
          cast_node.replace_input_with(cast_node, node)

    exported_program.graph_module.recompile()
    return fx_infra.ExportedProgramPassResult(exported_program, modified)
