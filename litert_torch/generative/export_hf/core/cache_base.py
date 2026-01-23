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
"""Base class for cache."""

import abc
from transformers import cache_utils


class LiteRTLMCacheLayerMixin(cache_utils.CacheLayerMixin, abc.ABC):
  """Optimized Cache layer class mixin for HuggingFace integration."""

  @abc.abstractmethod
  def get_batch_size(self) -> int:
    """Returns the batch size of the cache."""
    ...

  @classmethod
  @abc.abstractmethod
  def create_from_config(
      cls, model_config, layer_index, cache_length, batch_size=1, **kwargs
  ) -> "LiteRTLMCacheLayerMixin":
    ...


class LiteRTLMCacheMixin(cache_utils.Cache, abc.ABC):
  """Optimized Cache class mixin for HuggingFace integration."""

  @classmethod
  @abc.abstractmethod
  def create_from_config(
      cls, model_config, cache_length, batch_size=1
  ) -> "LiteRTLMCacheMixin":
    """Creates a KV cache from the model config."""
    ...


CACHE_REGISTRY: dict[str, type[LiteRTLMCacheMixin]] = {}


def register_cache_implementation(cls):
  """Decorator to register a cache implementation."""
  CACHE_REGISTRY[cls.__name__] = cls
  return cls
