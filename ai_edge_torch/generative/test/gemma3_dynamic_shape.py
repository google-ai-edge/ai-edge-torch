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
"""Export a toy Gemma2 model with dynamic prefill length."""

from absl import app
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import torch
from torch.export import Dim, export


# Define the wrapper class
class Gemma3Wrapper(torch.nn.Module):

  def __init__(self, model: gemma3.Gemma3MM):
    super().__init__()
    self.model = model
    self.config = model.config

  def forward(self, tokens, input_pos, mask, **kwargs):
    # Reconstruct the KVCache object from flattened kwargs
    num_layers = self.model.config.num_layers
    caches = []
    for i in range(num_layers):
      k_cache = kwargs[f"k_cache_{i}"]
      v_cache = kwargs[f"v_cache_{i}"]
      caches.append(kv_utils.KVCacheEntry(k_cache, v_cache))
    reconstructed_kv_cache = kv_utils.KVCache(caches)

    # Call the original model's forward pass
    return self.model(
        tokens=tokens,
        input_pos=input_pos,
        mask=mask,
        kv_cache=reconstructed_kv_cache,
    )


def main(_):
  # Export a fake Gemma2 model with a dynamic prefill length.
  original_model = gemma3.build_model_1b(checkpoint_path=None)
  # Wrap the model
  model = Gemma3Wrapper(original_model)

  # Create a dynamic KV sequence length, constrained by model config.
  max_seq_len = model.config.max_seq_len
  kv_cache_max_len = model.config.kv_cache_max_len
  print(f"max_seq_len: {max_seq_len}")
  print(f"kv_cache_max_len: {kv_cache_max_len}")

  # Tensors used to trace the model graph during conversion.
  # prefill_tokens = torch.full((1, 200), 0, dtype=torch.int)
  # prefill_input_pos = torch.arange(0, 200, dtype=torch.int)
  # Use max_seq_len for dimensions corresponding to dynamic dims.
  prefill_tokens = torch.full((1, max_seq_len), 0, dtype=torch.int)
  prefill_input_pos = torch.arange(0, max_seq_len, dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(model.config)

  kv_seq_len_dim = Dim("kv_seq_len", max=kv_cache_max_len)
  seq_len_dim = Dim("sequence_length", min=1, max=max_seq_len)

  # Define the dynamic shape spec for a single key/value tensor
  tensor_dynamic_spec = {1: kv_seq_len_dim}

  # Construct the dynamic shapes dictionary for flattened KV cache
  num_layers = model.config.num_layers
  dynamic_shapes = {
      "tokens": {1: seq_len_dim},
      "input_pos": {0: seq_len_dim},
      # Ensure the mask's dynamic dimensions align with the Dim definitions
      "mask": {2: seq_len_dim, 3: kv_seq_len_dim},
  }
  # Create a dictionary for the dynamic shapes of the kwargs
  kwargs_dynamic_shapes = {}
  for i in range(num_layers):
    kwargs_dynamic_shapes[f"k_cache_{i}"] = tensor_dynamic_spec
    kwargs_dynamic_shapes[f"v_cache_{i}"] = tensor_dynamic_spec
  # Add the kwargs dynamic shapes under the 'kwargs' key
  dynamic_shapes["kwargs"] = kwargs_dynamic_shapes

  print(f"dynamic_shapes: {dynamic_shapes}")
  # Use the max_seq_len for the mask dimension corresponding to kv_seq_len_dim
  # mask = torch.full((1, 1, 200, max_seq_len), float("-inf"), dtype=torch.float32)
  mask = torch.full((1, 1, max_seq_len, kv_cache_max_len), float("-inf"), dtype=torch.float32)
  torch._dynamo.mark_dynamic(mask, 2) # Corresponds to seq_len_dim
  torch._dynamo.mark_dynamic(mask, 3) # Corresponds to kv_seq_len_dim

  # Prepare flattened kwargs for the export call
  export_kwargs = {
      "tokens": prefill_tokens,
      "input_pos": prefill_input_pos,
      "mask": mask,
  }
  for i, cache_entry in enumerate(kv.caches):
    # Mark individual cache tensors as dynamic
    # Ensure the dimension marked dynamic corresponds to kv_seq_len_dim
    torch._dynamo.mark_dynamic(cache_entry.k_cache, 1)
    torch._dynamo.mark_dynamic(cache_entry.v_cache, 1)
    export_kwargs[f"k_cache_{i}"] = cache_entry.k_cache
    export_kwargs[f"v_cache_{i}"] = cache_entry.v_cache

  # # --- Manually create example KV cache tensors with correct kv_cache_max_len ---
  # batch_size = 1
  # # Ensure correct attribute access for Gemma3MM config structure
  # num_heads = model.config.block_configs[0].attn_config.num_heads
  # head_dim = model.config.block_configs[0].attn_config.head_dim
  # for i in range(model.config.num_layers):
  #   # Use kv_cache_max_len (2048) for the sequence dimension (dim 1)
  #   example_k_cache = torch.zeros((batch_size, kv_cache_max_len, num_heads, head_dim), dtype=torch.float32)
  #   example_v_cache = torch.zeros((batch_size, kv_cache_max_len, num_heads, head_dim), dtype=torch.float32)
  #   # Mark individual cache tensors as dynamic
  #   torch._dynamo.mark_dynamic(example_k_cache, 1) # Corresponds to kv_seq_len_dim
  #   torch._dynamo.mark_dynamic(example_v_cache, 1) # Corresponds to kv_seq_len_dim
  #   export_kwargs[f"k_cache_{i}"] = example_k_cache
  #   export_kwargs[f"v_cache_{i}"] = example_v_cache
  # # --- End manual creation ---

  # Export the wrapped model with flattened KV cache args
  ep = torch.export.export(
      model,
      args=(),
      kwargs=export_kwargs,
      dynamic_shapes=dynamic_shapes,
  )
  print(ep)


if __name__ == "__main__":
  app.run(main)
