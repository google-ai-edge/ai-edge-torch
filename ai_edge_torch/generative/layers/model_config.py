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
# Model configuration class.
from dataclasses import dataclass
from dataclasses import field
import enum
from typing import Optional


@enum.unique
class ActivationType(enum.Enum):
  """Different activation functions supported by the default builder."""

  LINEAR = enum.auto()
  SILU = enum.auto()
  GELU = enum.auto()
  GELU_TANH = enum.auto()
  GELU_QUICK = enum.auto()
  GE_GLU = enum.auto()
  RELU = enum.auto()


@enum.unique
class NormalizationType(enum.Enum):
  """Different normalization functions"""

  # No normalization is applied.
  NONE = enum.auto()
  RMS_NORM = enum.auto()
  LAYER_NORM = enum.auto()
  GROUP_NORM = enum.auto()


@enum.unique
class FeedForwardType(enum.Enum):
  """Different variations of the Feed Forward module."""

  # `output = linear(act(linear(x)))`.
  SEQUENTIAL = enum.auto()
  # `output = linear_2(act(linear_1(x)) * lienar_3(x))`.
  GATED = enum.auto()


@dataclass
class AttentionConfig:
  """Attention moduel's parameters."""

  num_heads: int
  # Used to determine number of groups in grouped query attention (GQA)
  # https://arxiv.org/pdf/2305.13245.pdf
  num_query_groups: Optional[int]
  # Percentage of Rotary Positional Embedding added Q and K projections.
  rotary_percentage: Optional[float] = None
  # Whether to transpose the query groups of qkv bundled tensor before
  # splitting into separated tensors.
  qkv_transpose_before_split: bool = False
  # Whether to use bias with Query, Key, and Value projection.
  qkv_use_bias: bool = False
  # Whether to use bias with attention output projection.
  output_proj_use_bias: bool = False
  enable_kv_cache: bool = True
  relative_attention_num_buckets: int = 0
  relative_attention_max_distance: int = 0


@dataclass
class ActivationConfig:
  type: ActivationType = ActivationType.LINEAR
  # Dimension of input and output, used in GeGLU.
  dim_in: Optional[int] = None
  dim_out: Optional[int] = None


@dataclass
class FeedForwardConfig:
  """FeedForward module's parameters."""

  type: FeedForwardType
  activation: ActivationConfig
  intermediate_size: int
  use_bias: bool = False


@dataclass
class NormalizationConfig:
  """Normalizater parameters."""

  type: NormalizationType = NormalizationType.NONE
  epsilon: float = 1e-5
  zero_centered: bool = False
  # Number of groups used in group normalization.
  group_num: Optional[float] = None


@dataclass
class ModelConfig:
  """Base configurations for building a transformer architecture."""

  vocab_size: int
  num_layers: int
  max_seq_len: int
  embedding_dim: int

  attn_config: AttentionConfig
  ff_config: FeedForwardConfig
  # The normalization applied to attention's input.
  pre_attention_norm_config: NormalizationConfig = field(
      default_factory=NormalizationConfig
  )
  # The normalization applied to feed forward's input.
  pre_ff_norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)
  # The normalization applied before LM head.
  final_norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)

  # If set to True, only pre_attention_norm is applied to the input and the
  # decode's output is computed as `output = input + attn_out + ff_out` where
  # attention and feed forward are called with pre_attention_norm's output.
  parallel_residual: bool = False
  # Use bias term within LLM's HEAD.
  lm_head_use_bias: bool = False
  # Whether to turn on high-level function boundary.
  enable_hlfb: bool = False

  # The maximum sequence length of the KV cache. Should not exceed max_seq_len.
  kv_cache_max_len: int = 0

  # The Attention computation will include relative positional bias.
  relative_attention: bool = False

  # Default batch size of the exported model. Default value is 1.
  batch_size: int = 1

  @property
  def kv_cache_max(self) -> int:
    if self.kv_cache_max_len > 0:
      return self.kv_cache_max_len
    else:
      return self.max_seq_len

  @property
  def head_dim(self) -> int:
    return self.embedding_dim // self.attn_config.num_heads
