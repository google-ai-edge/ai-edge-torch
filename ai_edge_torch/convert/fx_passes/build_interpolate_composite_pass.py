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

import functools

import torch

from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassBase
from ai_edge_torch.convert.fx_passes._pass_base import ExportedProgramPassResult  # NOQA
from ai_edge_torch.hlfb import mark_pattern

_interpolate_decompositions = torch._decomp.get_decompositions(
    [
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.upsample_nearest2d.vec,
    ]
)


@functools.cache
def _get_upsample_bilinear2d_pattern():
  pattern = mark_pattern.Pattern(
      "odml.upsample_bilinear2d",
      lambda x: torch.nn.functional.interpolate(
          x, scale_factor=2, mode="bilinear", align_corners=False
      ),
      export_args=(torch.rand(1, 3, 100, 100),),
  )

  @pattern.register_attr_builder
  def attr_builder(pattern, graph_module, internal_match):
    output = internal_match.returning_nodes[0]
    output_h, output_w = output.meta["val"].shape[-2:]
    return {
        "output": (int(output_h), int(output_w)),
        "align_corners": False,
    }

  pattern.run_decompositions(_interpolate_decompositions)
  return pattern


@functools.cache
def _get_upsample_bilinear2d_align_corners_pattern():
  pattern = mark_pattern.Pattern(
      "odml.upsample_bilinear2d",
      lambda x: torch.nn.functional.interpolate(
          x, scale_factor=2, mode="bilinear", align_corners=True
      ),
      export_args=(torch.rand(1, 3, 100, 100),),
  )

  @pattern.register_attr_builder
  def attr_builder(graph_module, pattern, internal_match):
    output = internal_match.returning_nodes[0]
    output_h, output_w = output.meta["val"].shape[-2:]
    return {
        "output": (int(output_h), int(output_w)),
        "align_corners": True,
    }

  pattern.run_decompositions(_interpolate_decompositions)
  return pattern


@functools.cache
def _get_interpolate_nearest2d_pattern():
  pattern = mark_pattern.Pattern(
      "tfl.resize_nearest_neighbor",
      lambda x: torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"),
      export_args=(torch.rand(1, 3, 100, 100),),
  )

  @pattern.register_attr_builder
  def attr_builder(pattern, graph_module, internal_match):
    output = internal_match.returning_nodes[0]
    output_h, output_w = output.meta["val"].shape[-2:]
    return {
        "size": (int(output_h), int(output_w)),
        "is_nchw_op": True,
    }

  pattern.run_decompositions(_interpolate_decompositions)
  return pattern


class BuildInterpolateCompositePass(ExportedProgramPassBase):

  def __init__(self):
    super().__init__()
    self._patterns = [
        _get_upsample_bilinear2d_pattern(),
        _get_upsample_bilinear2d_align_corners_pattern(),
        _get_interpolate_nearest2d_pattern(),
    ]

  def call(self, exported_program: torch.export.ExportedProgram):
    # Forward compatibility
    exported_program = exported_program.run_decompositions(_interpolate_decompositions)

    graph_module = exported_program.graph_module
    for pattern in self._patterns:
      graph_module = mark_pattern.mark_pattern(graph_module, pattern)

    graph_module.graph.lint()
    graph_module.recompile()
    return ExportedProgramPassResult(exported_program, True)
