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
"""Export a toy Gemma3 model with dynamic prefill length."""
import traceback

from absl import app
import ai_edge_torch
from ai_edge_torch import fx_infra
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.odml_torch.experimental import torch_tfl
import torch


# Define the wrapper class
class Gemma3Wrapper(torch.nn.Module):

  def __init__(self, model: gemma3.Gemma3MM):
    super().__init__()
    self.model = model
    self.config = model.config

  def forward(self, tokens, input_pos, mask,
              k_cache_0, v_cache_0, k_cache_1, v_cache_1, k_cache_2, v_cache_2,
              k_cache_3, v_cache_3, k_cache_4, v_cache_4, k_cache_5, v_cache_5,
              k_cache_6, v_cache_6, k_cache_7, v_cache_7, k_cache_8, v_cache_8,
              k_cache_9, v_cache_9, k_cache_10, v_cache_10, k_cache_11, v_cache_11,
              k_cache_12, v_cache_12, k_cache_13, v_cache_13, k_cache_14, v_cache_14,
              k_cache_15, v_cache_15, k_cache_16, v_cache_16, k_cache_17, v_cache_17,
              k_cache_18, v_cache_18, k_cache_19, v_cache_19, k_cache_20, v_cache_20,
              k_cache_21, v_cache_21, k_cache_22, v_cache_22, k_cache_23, v_cache_23,
              k_cache_24, v_cache_24, k_cache_25, v_cache_25):
    # Reconstruct the KVCache object from flattened kv_cache arguments
    num_layers = self.model.config.num_layers
    caches = []
    current_locals = locals()
    for i in range(num_layers):
      k_cache = current_locals[f"k_cache_{i}"]
      v_cache = current_locals[f"v_cache_{i}"]
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
  # Export a fake Gemma3 model with a dynamic prefill length.
  original_model = gemma3.build_model_1b(checkpoint_path=None)
  # Wrap the model
  model = Gemma3Wrapper(original_model)

  # Tensors used to trace the model graph during conversion.
  prefill_tokens = torch.full((1, 200), 0, dtype=torch.int)
  prefill_input_pos = torch.arange(0, 200, dtype=torch.int)
  kv = kv_utils.KVCache.from_model_config(2048, model.config)

  # Create a dynamic sequence length.
  seq_len_dim = torch.export.Dim("sequence_length", min=1, max=1024)
  # Create a dynamic sequence length for KV cache.
  kv_seq_len_dim = torch.export.Dim("kv_seq_len")

  # Define the dynamic shape spec for a single key/value tensor
  # This dictionary applies to both key_cache and value_cache tensors.
  tensor_dynamic_spec = {1: kv_seq_len_dim}

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
  # Merge kwargs_dynamic_shapes into the main dynamic_shapes to make it flat,
  # instead of nesting under a "kwargs" key.
  dynamic_shapes.update(kwargs_dynamic_shapes)
  print(f"dynamic_shapes: {dynamic_shapes}")

  mask = torch.full((1, 1, 200, 2048), float("-inf"), dtype=torch.float32)
  # Prepare a flat dictionary of all model inputs
  all_model_inputs_flat = {
      "tokens": prefill_tokens,
      "input_pos": prefill_input_pos,
      "mask": mask,
  }
  for i, cache_entry in enumerate(kv.caches):
    all_model_inputs_flat[f"k_cache_{i}"] = cache_entry.k_cache
    all_model_inputs_flat[f"v_cache_{i}"] = cache_entry.v_cache

  ep = torch.export.export(
      model,
      args=(),  # No positional arguments
      kwargs=all_model_inputs_flat,  # Pass all inputs as keyword arguments
      dynamic_shapes=dynamic_shapes, # dynamic_shapes is now flat
  )
  # print(f"Exported program: {ep}")

  # print("running model()")
  # original_model_output = model(**all_model_inputs_flat)
  # # Assuming the model returns a dictionary like
  # # {'logits': ..., 'kv_cache': ...}
  # original_logits = original_model_output["logits"]
  # print(f"original_model_output logits shape: {original_logits.shape}")

  # print("running ep.module()()")
  # exported_model_output = ep.module()(**all_model_inputs_flat)

  # # The exported model returns logits + updated flattened KV tensors
  # exported_logits = exported_model_output["logits"]
  # print(f"exported_model_output logits shape: {exported_logits.shape}")

  # # compare only the logits
  # assert torch.allclose(exported_logits, original_logits)
  # print("allclose(exported_logits, original_logits): True")

  try:
    print("Running TFL pre lower decompositions...")
    ep = fx_infra.safe_run_decompositions(
        ep, fx_infra.decomp.pre_lower_decomp()
    )
    print("TFL pre lower decompositions successful.")

    # print("running pre lower ep.module()()")
    # decomps_args, decomps_kwargs = ep.example_inputs
    # pre_lower_model_output = ep.module()(*decomps_args, **decomps_kwargs)
    # # pre_lower_model_output = ep.module()(**all_model_inputs_flat)

    # # The exported model returns logits + updated flattened KV tensors
    # pre_lower_logits = pre_lower_model_output["logits"]
    # print(f"pre_lower_model_output logits shape: {pre_lower_logits.shape}, dtype: {pre_lower_logits.dtype}")
    # print(f"original_logits shape: {original_logits.shape}, dtype: {original_logits.dtype}")

    # compare only the logits
    # Original assert was:
    # assert torch.allclose(pre_lower_logits, original_logits)

    # # Debugging the assertion failure:
    # are_close_by_default = torch.allclose(pre_lower_logits, original_logits)
    # if not are_close_by_default:
    #   print("\nDEBUG: torch.allclose(pre_lower_logits, original_logits) FAILED with default tolerances!")
    #   abs_diff = torch.abs(pre_lower_logits - original_logits)
    #   max_abs_diff = torch.max(abs_diff).item()
    #   print(f"  Max absolute difference: {max_abs_diff}")

    #   # Calculate relative difference carefully
    #   # Add a small epsilon to denominator for stability where original_logits might be zero
    #   epsilon = 1e-12
    #   # Ensure original_logits is float for division if it's not already
    #   rel_diff = abs_diff / (torch.abs(original_logits.float()) + epsilon)
    #   max_rel_diff = torch.max(rel_diff).item()
    #   print(f"  Max relative difference: {max_rel_diff}")

    #   # Count elements differing by more than a threshold (e.g., atol of allclose)
    #   significant_diff_threshold = 1e-5  # Example threshold
    #   num_significantly_differing_elements = torch.sum(abs_diff > significant_diff_threshold).item()
    #   print(f"  Number of elements with abs_diff > {significant_diff_threshold}: {num_significantly_differing_elements} (out of {pre_lower_logits.numel()})")

    #   # Show an example of a differing element (where abs_diff is max)
    #   if pre_lower_logits.numel() > 0:
    #     _, max_abs_diff_flat_idx = torch.max(abs_diff.view(-1), dim=0)
    #     max_abs_diff_idx_multidim = torch.unravel_index(max_abs_diff_flat_idx, pre_lower_logits.shape)
    #     print(f"  Example element with max absolute difference (at index {max_abs_diff_idx_multidim}):")
    #     print(f"    Original value: {original_logits[max_abs_diff_idx_multidim].item()}")
    #     print(f"    Pre-lower value: {pre_lower_logits[max_abs_diff_idx_multidim].item()}")
    #     print(f"    Difference: {abs_diff[max_abs_diff_idx_multidim].item()}")

    #   print("  Consider checking with custom tolerances if differences are small, e.g.:")
    #   print("    torch.allclose(pre_lower_logits, original_logits, rtol=1e-4, atol=1e-5)\n")

    # # The script will still fail here if they are not close by default tolerances,
    # # but now you have more diagnostic information printed above.
    # assert torch.allclose(pre_lower_logits, original_logits), \
    #     "Logits mismatch after pre_lower decompositions. See DEBUG printouts above for details."
    # print("allclose(pre_lower_logits, original_logits): True")
    # print("------------------------------------------")
  except Exception as e:
    print(f"Error during pre lower decompositions: {e}")
    print(f"Traceback during pre lower ep.module: {traceback.format_exc()}")

  # Initialize decomps_ep outside the try block
  decomps_ep = None
  # Now, run the TFL decompositions (where the error originally occurred)
  try:
    print("Running TFL decompositions...")
    decomps_ep = ep.run_decompositions(torch_tfl.decomps)
    print("TFL decompositions successful.")
    # print(f"TFL decomps ep: {decomps_ep}")
    # print(f"TFL decomps graph: {decomps_ep.graph}")
    print("------------------------------------------")

    print("--- Nodes with Symbolic Dimensions (Post-Decomposition) ---")

    def has_symbolic_dim(val):
      """Checks if a value (like from node.meta['val']) has a symbolic shape."""
      # Case 1: The value itself is a symbolic integer.
      if isinstance(val, torch.SymInt):
        return True

      # Case 2: The value is a tensor-like object with a shape.
      # Exported tensor metadata usually has a .shape of type torch.Size.
      if hasattr(val, "shape") and isinstance(val.shape, torch.Size):
        for dim_element in val.shape:  # Iterate directly over torch.Size elements
          if isinstance(dim_element, torch.SymInt):
            return True
        # If the loop completes, no symbolic dimension was found in this shape.
        return False

      # Case 3: The value is a list or tuple (e.g., for multi-output nodes).
      # Recursively check each item.
      if isinstance(val, (list, tuple)):
        for item in val:
          if has_symbolic_dim(item):
            return True
        # If the loop completes, no symbolic item was found in the list/tuple.
        return False

      # If none of the above, it's not considered symbolic by this function.
      return False

    nodes_with_symbolic_dims_found = False
    node_targets_with_symbolic_dims = set()
    count_node_with_symbolic_dims = 0
    # Iterate through the nodes of the *decomposed* graph
    for node in decomps_ep.graph.nodes:
      symbolic_found = False
      # Check node output's symbolic nature
      output_val = node.meta.get("val")  # 'val' is common in FX meta
      if has_symbolic_dim(output_val):
        symbolic_found = True

      # Check node inputs' symbolic nature (if output wasn't symbolic)
      if not symbolic_found:
        # Combine args and kwargs values for checking
        all_args = list(node.args) + list(node.kwargs.values())
        for arg in all_args:
          if isinstance(arg, torch.fx.Node):
            # Check the output shape of the input node
            input_val = arg.meta.get("val")
            if has_symbolic_dim(input_val):
              symbolic_found = True
              break  # Found symbolic dim in inputs
          elif has_symbolic_dim(arg):  # Check if arg itself is symbolic
            symbolic_found = True
            break

      if symbolic_found:
        nodes_with_symbolic_dims_found = True
        node_targets_with_symbolic_dims.add(str(node.target))
        count_node_with_symbolic_dims += 1

        # print(f"Node: {node.op} target={node.target} | name={node.name}")
        # print(f"  Output Meta['val']: {output_val}")
        # # Optionally print args/kwargs for more context
        # print(f"  Args: {node.args}")
        # print(f"  Kwargs: {node.kwargs}")
        # print("-" * 20)

    if not nodes_with_symbolic_dims_found:
      print("No nodes with symbolic dimensions found in the decomposed graph.")
    print(
        "Unique node targets with symbolic dims:"
        f" {node_targets_with_symbolic_dims}"
    )
    print(f"Count of nodes with symbolic dims: {count_node_with_symbolic_dims}")
    print(f"Total nodes: {len(decomps_ep.graph.nodes)}")
    print(f"Percent of nodes with symbolic dims: {count_node_with_symbolic_dims / len(decomps_ep.graph.nodes) * 100:.2f}")
    print("---------------------------------------------------------")
  except torch._export.verifier.SpecViolationError as e:
    print(f"SpecViolationError during TFL decompositions: {e}")
  except Exception as e:
    print(f"Error during TFL decompositions: {e}")
    print(f"Traceback during TFL decompositions: {traceback.format_exc()}")

  # try:
  #   decomps_args, decomps_kwargs = decomps_ep.example_inputs
  #   decomps_model_output = decomps_ep.module()(*decomps_args, **decomps_kwargs)
  #   decomps_logits = decomps_model_output["logits"]
  #   print(f"decomps_model_output logits shape: {decomps_logits.shape}")

  #   # compare only the logits
  #   assert torch.allclose(decomps_logits, original_logits)
  #   print("allclose(decomps_logits, original_logits): True")
  #   print("------------------------------------------")
  # except Exception as e:
  #   print(f"Error during decomps_ep.module: {e}")
  #   print(f"Traceback during decomps_ep.module: {traceback.format_exc()}")

  try:
    print("Running ai_edge_torch.convert...")

    edge_model = ai_edge_torch.convert(
        decomps_ep.module(),
        sample_args=(),
        sample_kwargs=all_model_inputs_flat,
        dynamic_shapes=dynamic_shapes,
    )
    print("ai_edge_torch.convert successful.")
    print(f"Converted model: {edge_model}")
  except Exception as e:
    print(f"Error during ai_edge_torch.convert: {e}")

if __name__ == "__main__":
  app.run(main)
