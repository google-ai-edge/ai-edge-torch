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

"""Example of building the Moonshine model."""

import os
import pathlib
from typing import Optional, Tuple
from absl import app
from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.normalization as normalization
import ai_edge_torch.generative.utilities.moonshine_loader as loading_utils
import h5py
import torch
from torch import nn
import torch.nn as nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    conv1D_0="layers/sequential/layers/conv1d/vars",
    conv1D_1="layers/sequential/layers/conv1d_1/vars",
    conv1D_2="layers/sequential/layers/conv1d_2/vars",
    group_norm="layers/sequential/layers/group_normalization/vars",
)


class AudioPreprocessor(nn.Module):

  def __init__(self, dim):
    super(AudioPreprocessor, self).__init__()
    self.conv1 = nn.Conv1d(
        in_channels=1, out_channels=dim, kernel_size=127, stride=64, bias=False
    )
    self.tanh = nn.Tanh()
    self.group_norm = normalization.GroupNorm(group_num=1, dim=dim, eps=1e-5)
    self.conv2 = nn.Conv1d(
        in_channels=dim,
        out_channels=2 * dim,
        kernel_size=7,
        stride=3,
        padding=0,  # Equivalent to padding="valid"
    )
    self.gelu1 = nn.GELU()
    self.conv3 = nn.Conv1d(
        in_channels=2 * dim,
        out_channels=dim,
        kernel_size=3,
        stride=2,
        padding=0,  # Equivalent to padding="valid"
    )
    self.gelu2 = nn.GELU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.tanh(x)
    x = self.group_norm(x)
    x = self.conv2(x)
    x = self.gelu1(x)
    x = self.conv3(x)
    x = self.gelu2(x)
    return x


def build_preprocessor(checkpoint_path: str, **kwargs) -> nn.Module:
  ap = AudioPreprocessor(dim=416)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(ap, strict=True)
  return ap


def main(_):
  # TODO(b/375421767) Remove golden checks once full model is implemented.
  HF_PATH = os.path.join(pathlib.Path.home(), "Downloads/llm_data/moonshine")

  test_data_path = pathlib.Path(__file__).parent.resolve()
  INPUT_PATH = test_data_path / "data" / "pp_input.pt")
  GOLDEN_PATH = test_data_path / "data" / "pp_output.pt")

  ap = build_preprocessor(HF_PATH)
  ap.eval()
  inputs = torch.load(INPUT_PATH).reshape((1, 1, 159414))
  out = ap(inputs)
  golden = torch.load(GOLDEN_PATH).transpose(1, 2)
  assert torch.allclose(out, golden, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
  app.run(main)
