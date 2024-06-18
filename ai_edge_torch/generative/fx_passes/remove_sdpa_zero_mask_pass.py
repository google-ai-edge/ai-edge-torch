import torch

from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassBase
from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassResult  # NOQA


class RemoveSDPACompositeZeroMaskPass(ExportedProgramPassBase):

  def is_zero_tensor_node(self, node: torch.fx.Node):
    return node.target == torch.ops.aten.zeros.default

  def call(self, exported_program: torch.export.ExportedProgram):
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
      if not (
          node.op == "call_function"
          and node.target == torch.ops.xla.mark_tensor.default
      ):
        continue

      source, name, io_num, id, is_input = node.args[:5]
      # Composite info:
      # - name: odml.scaled_dot_product_attention
      # - inputs: q, k, v, mask
      if name == "odml.scaled_dot_product_attention" and is_input and io_num == 3:
        if self.is_zero_tensor_node(source):
          # Remove the mark_tensor call on the mask input by
          # replacing the target with a identity function
          node.target = lambda *args, **kwargs: args[0]

    exported_program.graph_module.graph.lint()
    exported_program.graph_module.recompile()
    return ExportedProgramPassResult(exported_program, True)
