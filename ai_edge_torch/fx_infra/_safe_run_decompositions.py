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
"""ExportedProgram.run_decompositions wrapper to handle unexpected export behavior."""
import torch


# A dummy decomp table for running ExportedProgram.run_decompositions without
# any op decompositions but just aot_export_module. Due to the check in
# run_decompositions, if None or an empty dict is passed as decomp_table,
# it will run the default aten-coreaten decompositions. Therefore a non-empty
# dummy decomp table is needed.
# Ref: https://github.com/pytorch/pytorch/blob/db895ace1d36726e64781774f53b3d3098206116/torch/export/exported_program.py#L543
_DUMMY_DECOMP_TABLE = {
    torch._ops.OperatorBase(): lambda: None,
}


def safe_run_decompositions(exported_program, decomp_table=None):
  """Wrapper for ExportedProgram.run_decompositions to handle unexpected export behavior."""

  if decomp_table is not None and not decomp_table:
    # Empty decomp table means no op decompositions. Use dummy decomp table
    # instead for backward compatibility.
    decomp_table = _DUMMY_DECOMP_TABLE

  for node in exported_program.graph.nodes:
    if node.target == torch.ops.aten.view.default:
      # Passes or torch.export may generate aten.view nodes not respecting the
      # tensor memory format. Changes all the aten.view to torch.reshape
      # for retracing. If the input memory format is already contiguous,
      # retracing in run_decomposition below would decompose torch.reshape
      # back to one aten.view.
      node.target = lambda self, size: torch.reshape(self.contiguous(), size)

  exported_program = exported_program.run_decompositions(decomp_table)

  if hasattr(torch.ops.aten, "_assert_tensor_metadata"):
    for node in exported_program.graph.nodes:
      if node.target == torch.ops.aten._assert_tensor_metadata.default:
        exported_program.graph.erase_node(node)

  exported_program.graph.eliminate_dead_code()
  exported_program.graph_module.recompile()

  return exported_program
