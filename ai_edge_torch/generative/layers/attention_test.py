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

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import model_config as cfg
import torch

from absl.testing import absltest as googletest
from absl.testing import parameterized


class AttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="local_causal_self_attention",
          attn_type=cfg.AttentionType.LOCAL_SLIDING,
          use_alibi=False,
          expected_shape=(1, 10, 16),
      ),
      dict(
          testcase_name="global_causal_self_attention",
          attn_type=cfg.AttentionType.GLOBAL,
          use_alibi=False,
          expected_shape=(1, 10, 16),
      ),
      dict(
          testcase_name="alibi_attention",
          attn_type=cfg.AttentionType.GLOBAL,
          use_alibi=True,
          expected_shape=(1, 10, 16),
      ),
  )
  def test_causal_self_attention(
      self,
      attn_type: cfg.AttentionType,
      use_alibi: bool,
      expected_shape: tuple[int, ...],
  ):
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=1e-6,
        zero_centered=True,
        enable_hlfb=True,
    )
    attn_config = cfg.AttentionConfig(
        num_heads=2,
        head_dim=8,
        num_query_groups=1,
        rotary_base=100,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=16,
        attn_type=attn_type,
        use_alibi=use_alibi,
    )
    self_atten = attention.CausalSelfAttention(
        dim=16,
        config=attn_config,
        enable_hlfb=True,
    )
    x = torch.randn(1, 10, 16)
    attn_mask = torch.ones((1, 1, 10, 10), dtype=torch.float32)
    out = self_atten(x, rope=None, mask=attn_mask)
    self.assertEqual(out.shape, expected_shape)

  def test_cross_attention(self):
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=1e-6,
        zero_centered=True,
        enable_hlfb=True,
    )
    attn_config = cfg.AttentionConfig(
        num_heads=2,
        head_dim=8,
        num_query_groups=1,
        rotary_base=100,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=16,
        attn_type=cfg.AttentionType.GLOBAL,
    )
    cross_atten = attention.CrossAttention(
        query_dim=16,
        cross_dim=16,
        hidden_dim=16,
        output_dim=16,
        config=attn_config,
        enable_hlfb=True,
    )
    x = torch.randn(1, 10, 16)
    y = torch.randn(1, 10, 16)
    out = cross_atten(x, y, rope=None)
    self.assertEqual(out.shape, (1, 10, 16))

  def test_transformer_block(self):
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=1e-6,
        zero_centered=True,
        enable_hlfb=True,
    )
    attn_config = cfg.AttentionConfig(
        num_heads=2,
        head_dim=8,
        num_query_groups=1,
        rotary_base=100,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=16,
        attn_type=cfg.AttentionType.GLOBAL,
    )
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.GATED,
        activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
        intermediate_size=32,
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        post_attention_norm_config=norm_config,
        parallel_residual=True,
    )
    model_config = cfg.ModelConfig(
        vocab_size=100,
        embedding_dim=16,
        enable_hlfb=True,
        num_layers=1,
        max_seq_len=10,
        block_configs=[block_config],
    )
    transformer_block = attention.TransformerBlock(
        config=block_config,
        model_config=model_config,
    )
    x = torch.randn(1, 10, 16)
    attn_mask = torch.ones((1, 1, 10, 10), dtype=torch.float32)
    out = transformer_block(x, rope=None, mask=attn_mask)
    self.assertEqual(out.shape, (1, 10, 16))


if __name__ == "__main__":
  googletest.main()
