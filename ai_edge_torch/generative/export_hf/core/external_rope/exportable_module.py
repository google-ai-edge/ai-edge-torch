# Copyright 2026 The AI Edge Torch Authors.
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
"""Exportable module for externalized rotary embedding."""

from ai_edge_torch.generative.export_hf.core import exportable_module as base_exportable_module
import torch


class RoPEEmbedder(torch.nn.Module):
  """Exportable module for decoder only LM running on NPU."""

  def __init__(self, model):
    super().__init__()
    self.model = model

    assert self.model.model.original_rotary_emb is not None
    self.rotary_emb = self.model.model.original_rotary_emb
    if hasattr(self.model.model, 'original_rotary_emb_local'):
      self.rotary_emb_local = self.model.model.original_rotary_emb_local
    else:
      self.rotary_emb_local = None

  def forward(
      self,
      input_pos: torch.Tensor,  # [T]
  ):
    dummy = torch.ones((1, 1, 1), dtype=torch.float32)
    position_ids = input_pos.unsqueeze(0)
    pos_emb = self.rotary_emb(dummy, position_ids)
    ret = {
        'pos_emb_cos': pos_emb[0],
        'pos_emb_sin': pos_emb[1],
    }
    if self.rotary_emb_local is not None:
      pos_emb_local = self.rotary_emb_local(dummy, position_ids)
      ret.update({
          'pos_emb_local_cos': pos_emb_local[0],
          'pos_emb_local_sin': pos_emb_local[1],
      })
    return ret

  @classmethod
  def get_sample_inputs(
      cls,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
  ):
    """Gets sample inputs."""
    del model_config
    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      inputs = {
          'input_pos': torch.arange(prefill_length, dtype=torch.int32),
      }
      sample_inputs[f'prefill_rope_{prefill_length}'] = (inputs, {})
    decode_inputs = {
        'input_pos': torch.arange(1, dtype=torch.int32),
    }
    sample_inputs['decode_rope'] = (decode_inputs, {})
    return sample_inputs
