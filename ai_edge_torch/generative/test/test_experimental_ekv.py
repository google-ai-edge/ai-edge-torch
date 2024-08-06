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
# A suite of tests to validate experimental external KV Cache layers and models.

from ai_edge_torch.generative.examples.experimental.gemma import gemma
from ai_edge_torch.generative.examples.experimental.phi import phi2
from ai_edge_torch.generative.examples.experimental.tiny_llama import tiny_llama  # NOQA
from ai_edge_torch.generative.layers.experimental import ekv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
import torch

from tensorflow.python.platform import googletest


class TestExternalKVLayers(googletest.TestCase):

  def _get_test_config(
      self, num_layers, head_dim, num_query_groups, kv_cache_max_len
  ):
    attn_config = cfg.AttentionConfig(
        num_heads=1, head_dim=head_dim, num_query_groups=num_query_groups
    )
    config = cfg.ModelConfig(
        kv_cache_max_len=kv_cache_max_len,
        embedding_dim=head_dim,
        attn_config=attn_config,
        num_layers=num_layers,
        max_seq_len=None,
        vocab_size=None,
        ff_config=None,
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
    kv = kv_utils.EKVCache.from_model_config(config)
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
    input_pos = torch.tensor([0, 3])
    k_slice = v_slice = torch.full(
        (1, 2, NUM_QG, HEAD_DIM), 7, dtype=torch.float
    )
    updated_entry = kv_utils.update(entry, input_pos, k_slice, v_slice)
    self.assertEqual(
        updated_entry.k_cache.numpy().flatten().tolist(),
        [7, 7, 0, 0, 0, 0, 7, 7],
    )
    self.assertEqual(
        updated_entry.v_cache.numpy().flatten().tolist(),
        [7, 7, 0, 0, 0, 0, 7, 7],
    )

  def test_serialization(self):
    class TestModel(torch.nn.Module):

      def forward(self, kv: kv_utils.EKVCache) -> kv_utils.EKVCache:
        updated_kv_entries = [
            kv_utils.KVCacheEntry(
                torch.zeros_like(entry.k_cache), torch.zeros_like(entry.v_cache)
            )
            for entry in kv.caches
        ]
        return kv_utils.EKVCache(updated_kv_entries)

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
    kv = kv_utils.EKVCache.from_model_config(config)
    model = TestModel()
    exported_program = torch.export.export(model, (kv,))
    input_specs = exported_program.graph_signature.input_specs
    self.assertEqual(len(input_specs), 2)
    self.assertEqual(input_specs[0].arg.name, "kv_k_0")
    self.assertEqual(input_specs[1].arg.name, "kv_v_0")


class TestExternalKVModels(googletest.TestCase):

  def test_can_build_gemma(self):
    gemma.define_and_run_2b(checkpoint_path=None, test_model=True)

  def test_can_build_phi2(self):
    phi2.define_and_run(checkpoint_path=None, test_model=True)

  def test_can_build_tinyllama(self):
    tiny_llama.define_and_run(checkpoint_path=None, test_model=True)


if __name__ == "__main__":
  googletest.main()
