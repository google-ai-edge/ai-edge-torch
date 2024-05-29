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

import torch
from torch import nn
from torch._prims_common import mask_tensor
from torch._prims_common.wrappers import out_wrapper

from ai_edge_torch.generative.examples.stable_diffusion.attention import SelfAttention  # NOQA


class CLIPEmbedding(nn.Module):

  def __init__(self, n_vocab: int, n_embd: int, n_token: int):
    super().__init__()
    self.token_embedding = nn.Embedding(n_vocab, n_embd)
    self.position_value = nn.Parameter(torch.zeros((n_token, n_embd)))

  def forward(self, tokens):
    x = self.token_embedding(tokens)
    x += self.position_value
    return x


class CLIPLayer(nn.Module):

  def __init__(self, n_head: int, n_embd: int):
    super().__init__()
    self.layernorm_1 = nn.LayerNorm(n_embd)
    self.attention = SelfAttention(n_head, n_embd)
    self.layernorm_2 = nn.LayerNorm(n_embd)
    self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
    self.linear_2 = nn.Linear(4 * n_embd, n_embd)

  def forward(self, x):
    residue = x
    x = self.layernorm_1(x)
    x = self.attention(x, causal_mask=True)
    x += residue

    residue = x
    x = self.layernorm_2(x)
    x = self.linear_1(x)
    x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
    x = self.linear_2(x)
    x += residue

    return x


class CLIP(nn.Module):

  def __init__(self):
    super().__init__()
    self.embedding = CLIPEmbedding(49408, 768, 77)
    self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])
    self.layernorm = nn.LayerNorm(768)

  @torch.inference_mode
  def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
    tokens = tokens.type(torch.long)

    state = self.embedding(tokens)
    for layer in self.layers:
      state = layer(state)
    output = self.layernorm(state)
    return output
