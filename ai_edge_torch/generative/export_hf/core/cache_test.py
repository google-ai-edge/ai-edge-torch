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
"""Tests for cache layers."""

from typing import List, Tuple

from ai_edge_torch.generative.export_hf.core import cache as cache_lib
import torch

from absl.testing import absltest as googletest


def build_cache_data(
    batch_size: int,
    num_layers: int,
    context_len: int,
    head_dim: int,
    all_ones: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
  cache_data = []
  for _ in range(num_layers):
    if all_ones:
      key_cache = torch.ones(
          1, batch_size, context_len, head_dim, dtype=torch.float32
      )
      value_cache = torch.ones(
          1, batch_size, head_dim, context_len, dtype=torch.float32
      )
    else:
      key_cache = torch.randn(
          1, batch_size, context_len, head_dim, dtype=torch.float32
      )
      value_cache = torch.randn(
          1, batch_size, head_dim, context_len, dtype=torch.float32
      )
    cache_data.append(cache_lib.LiteRTLMCacheLayer(key_cache, value_cache))
  return cache_data


def update_cache(slices, kv_cache, cache_kwargs):
  """Updates the cache with the given slices."""
  for i in range(len(slices)):
    kv_cache.update(slices[i][0], slices[i][1], i, cache_kwargs)
  return kv_cache


class CacheTest(googletest.TestCase):

  def assert_cache_equals(
      self,
      kv_cache: cache_lib.LiteRTLMCache,
      expected_kv_cache: cache_lib.LiteRTLMCache,
      num_layers,
  ):
    """Asserts that the cache is equal to the expected cache."""
    for i in range(num_layers):
      layer_cache = kv_cache.layers[i]
      expected_layer_cache = expected_kv_cache.layers[i]
      self.assertTrue(
          torch.allclose(layer_cache.keys, expected_layer_cache.keys)
      )
      self.assertTrue(
          torch.allclose(layer_cache.values, expected_layer_cache.values)
      )

  def assert_cache_slice_equals(
      self,
      slices: List[Tuple[torch.Tensor, torch.Tensor]],
      kv_cache: cache_lib.LiteRTLMCache,
      num_layers,
      time_step,
      input_seq,
  ):
    """Asserts that the cache slices are equal to the expected slices."""
    for i in range(num_layers):
      k_slice = kv_cache.layers[i].keys[
          :, :, time_step : time_step + input_seq, :
      ]
      v_slice = kv_cache.layers[i].values[
          :, :, :, time_step : time_step + input_seq
      ]
      expected_k_slice = slices[i][0]
      expected_v_slice = slices[i][1]
      self.assertTrue(torch.allclose(k_slice, expected_k_slice))
      self.assertTrue(torch.allclose(v_slice, expected_v_slice))

  def test_accessors(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64

    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    # Cache entries shape.
    self.assertEqual(
        kv_cache.layers[0].keys.shape,
        (1, batch_head_size, context_len, head_dim),
    )
    self.assertEqual(
        kv_cache.layers[0].values.shape,
        (1, batch_head_size, head_dim, context_len),
    )
    self.assertLen(kv_cache.layers, num_layers)
    # Cache attributes
    self.assertTrue(kv_cache.is_compileable)
    self.assertTrue([not x for x in kv_cache.is_sliding])
    self.assertEqual(kv_cache.max_cache_len, context_len)

  def test_update(self):
    batch_size = 2
    batch_head_size = 8
    kv_head_size = batch_head_size // batch_size
    num_layers = 5
    context_len = 1024
    head_dim = 64
    input_seq = 10
    time_step = 33

    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(
            batch_head_size, num_layers, context_len, head_dim, all_ones=True
        )
    )
    cache_kwargs = {
        "cache_position": torch.tensor([time_step], dtype=torch.int32)
    }
    k_slice = torch.zeros(batch_size, kv_head_size, input_seq, head_dim)
    v_slice = torch.zeros(batch_size, kv_head_size, input_seq, head_dim)
    slices = [(k_slice, v_slice)] * num_layers
    expected_k_slice = torch.zeros(
        1, batch_size * kv_head_size, input_seq, head_dim
    )
    expected_v_slice = torch.zeros(
        1, batch_size * kv_head_size, head_dim, input_seq
    )
    expected_slices = [(expected_k_slice, expected_v_slice)] * num_layers

    kv_cache = update_cache(slices, kv_cache, cache_kwargs)

    self.assert_cache_slice_equals(
        expected_slices, kv_cache, num_layers, time_step, input_seq
    )

  def test_flatten_round_trip(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64
    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    flattened, context = cache_lib._flatten_kvc_t(kv_cache)
    unflattened = cache_lib._unflatten_kvc_t(flattened, context)

    self.assertLen(flattened, num_layers * 2)
    self.assertEqual(
        kv_cache.layers[0].keys.shape, unflattened.layers[0].keys.shape
    )
    self.assertEqual(
        kv_cache.layers[0].values.shape, unflattened.layers[0].values.shape
    )
    self.assert_cache_equals(kv_cache, unflattened, num_layers)

  def test_flatten_with_keys(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64
    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    flattened_list, flattend_names = cache_lib._flatten_kvc_t_with_keys(
        kv_cache
    )
    self.assertLen(flattened_list, num_layers * 2)
    self.assertLen(flattend_names, num_layers * 2)


if __name__ == "__main__":
  googletest.main()
