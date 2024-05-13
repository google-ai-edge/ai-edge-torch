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
import contextlib
import sys

import torch
from torch.export.graph_signature import InputKind
import torch.fx._pytree as fx_pytree
from torch.utils import _pytree as pytree


def exported_program_to_fx_graph_module_and_inputs(ep: torch.export.ExportedProgram):
  fx_gm = ep.graph_module
  fx_inputs = pytree.tree_map(
      torch.tensor, ep._graph_module_flat_inputs(*ep.example_inputs)
  )
  return fx_gm, fx_inputs


@contextlib.contextmanager
def redirect_stdio(stdout, stderr):
  old_stdout = sys.stdout
  old_stderr = sys.stderr

  old_stdout.flush()
  old_stderr.flush()

  sys.stdout = stdout
  sys.stderr = stderr
  try:
    yield stdout, stderr
  finally:
    stdout.flush()
    stderr.flush()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
