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

"""A suite of tests to validate KV Cache layer."""

from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers.experimental import kv_cache as kv_utils_experimental
import ai_edge_torch.generative.layers.model_config as cfg
import torch
import torch.utils._pytree as pytree

from absl.testing import absltest as googletest


class TestKVLayers(googletest.TestCase):

  def _get_test_config(
      self, num_layers, head_dim, num_query_groups, kv_cache_max_len
  ):
    attn_config = cfg.AttentionConfig(
        num_heads=1, head_dim=head_dim, num_query_groups=num_query_groups
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config, ff_config=None
    )
    config = cfg.ModelConfig(
        kv_cache_max_len=kv_cache_max_len,
        embedding_dim=head_dim,
        block_configs=block_config,
        num_layers=num_layers,
        max_seq_len=None,
        vocab_size=None,
    )
    return config

  def test_cache_udpate(self):
    N = 1
    HEAD_DIM = 2
    NUM_QG = 1
    KV_LEN = 4
    config = self._get_test_config(
        num_layers=N,
        head_dim=HEAD_DIM,
        num_query_groups=NUM_QG,
        kv_cache_max_len=KV_LEN,
    )
    kv = kv_utils.KVCache.from_model_config(config)
    entry = kv.caches[0]
    # single-slice update
    input_pos = torch.tensor([1])
    k_slice = v_slice = torch.full(
        (1, 1, NUM_QG, HEAD_DIM), 5, dtype=torch.float
    )
    updated_entry = kv_utils.update(entry, input_pos, k_slice, v_slice)
    self.assertEqual(
        updated_entry.k_cache.numpy().flatten().tolist(),
        [0, 0, 5, 5, 0, 0, 0, 0],
    )
    self.assertEqual(
        updated_entry.v_cache.numpy().flatten().tolist(),
        [0, 0, 5, 5, 0, 0, 0, 0],
    )
    # multi-slice update
    input_pos = torch.tensor([0, 1])
    k_slice = v_slice = torch.full(
        (1, 2, NUM_QG, HEAD_DIM), 7, dtype=torch.float
    )
    updated_entry = kv_utils.update(entry, input_pos, k_slice, v_slice)
    self.assertEqual(
        updated_entry.k_cache.numpy().flatten().tolist(),
        [7, 7, 7, 7, 0, 0, 0, 0],
    )
    self.assertEqual(
        updated_entry.v_cache.numpy().flatten().tolist(),
        [7, 7, 7, 7, 0, 0, 0, 0],
    )

  def test_serialization(self):
    class TestModel(torch.nn.Module):

      def forward(self, kv: kv_utils.KVCache) -> kv_utils.KVCache:
        updated_kv_entries = [
            kv_utils.KVCacheEntry(
                torch.zeros_like(entry.k_cache), torch.zeros_like(entry.v_cache)
            )
            for entry in kv.caches
        ]
        return kv_utils.KVCache(updated_kv_entries)

    N = 1
    HEAD_DIM = 2
    NUM_QG = 1
    KV_LEN = 4
    config = self._get_test_config(
        num_layers=N,
        head_dim=HEAD_DIM,
        num_query_groups=NUM_QG,
        kv_cache_max_len=KV_LEN,
    )
    kv = kv_utils.KVCache.from_model_config(config)
    model = TestModel()
    exported_program = torch.export.export(model, (kv,))
    input_specs = exported_program.graph_signature.input_specs
    self.assertEqual(len(input_specs), 2)
    self.assertEqual(input_specs[0].arg.name, "kv_k_0")
    self.assertEqual(input_specs[1].arg.name, "kv_v_0")

  def test_pytree_roundtrip_experimental_kv_cache_base(self):
    NUM_LAYERS = 4
    config = self._get_test_config(
        num_layers=NUM_LAYERS,
        head_dim=2,
        num_query_groups=1,
        kv_cache_max_len=4,
    )
    kv = kv_utils_experimental.KVCacheBase.from_model_config(
        config, batch_size=1
    )
    flat, treespec = pytree.tree_flatten(kv)
    self.assertLen(flat, NUM_LAYERS * 2)
    kv_unflat = pytree.tree_unflatten(flat, treespec)
    self.assertEqual(kv, kv_unflat)

  def test_pytree_roundtrip_experimental_kv_cache_derived(self):
    NUM_LAYERS = 4
    config = self._get_test_config(
        num_layers=NUM_LAYERS,
        head_dim=2,
        num_query_groups=1,
        kv_cache_max_len=4,
    )
    kv = kv_utils_experimental.KVCacheTransposed.from_model_config(
        config, batch_size=1
    )
    flat, treespec = pytree.tree_flatten(kv)
    self.assertLen(flat, NUM_LAYERS * 2)
    kv_unflat = pytree.tree_unflatten(flat, treespec)
    self.assertEqual(kv, kv_unflat)

  def test_pytree_roundtrip_experimental_kv_entry_base(self):
    attn_config = cfg.AttentionConfig(
        num_heads=1, head_dim=1, num_query_groups=1
    )
    kv = kv_utils_experimental.KVCacheEntryBase.from_model_config(
        32, attn_config
    )
    flat, treespec = pytree.tree_flatten(kv)
    self.assertLen(flat, 2)
    kv_unflat = pytree.tree_unflatten(flat, treespec)
    self.assertEqual(kv, kv_unflat)
    self.assertIsInstance(kv_unflat, kv_utils_experimental.KVCacheEntryBase)

  def test_pytree_roundtrip_experimental_kv_entry_derived(self):
    attn_config = cfg.AttentionConfig(
        num_heads=1, head_dim=1, num_query_groups=1
    )
    kv = kv_utils_experimental.KVCacheEntryTransposed.from_model_config(
        32, attn_config
    )
    flat, treespec = pytree.tree_flatten(kv)
    self.assertLen(flat, 2)
    kv_unflat = pytree.tree_unflatten(flat, treespec)
    self.assertEqual(kv, kv_unflat)
    self.assertIsInstance(
        kv_unflat, kv_utils_experimental.KVCacheEntryTransposed
    )


if __name__ == "__main__":
  googletest.main()
