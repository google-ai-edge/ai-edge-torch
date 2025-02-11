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

"""Common utility functions for model conversion."""

import os
from typing import Any, Optional, Union

import ai_edge_torch._convert.converter as converter_utils
import ai_edge_torch.generative.layers.kv_cache as kv_utils
import ai_edge_torch.generative.layers.lora as lora_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.quantize import quant_recipes
import torch


class ExportableModule(torch.nn.Module):
  """A wrapper for a model to convert to tflite with non-exportable args."""

  def __init__(self, module, **extra_kwargs):
    super().__init__()
    self.module = module
    self.extra_kwargs = extra_kwargs

  def forward(self, *export_args, **export_kwargs):
    full_kwargs = {**export_kwargs, **self.extra_kwargs}
    return self.module(*export_args, **full_kwargs)


def convert_to_tflite(
    pytorch_model: torch.nn.Module,
    output_path: str,
    output_name_prefix: str,
    prefill_seq_len: Union[int, list[int]],
    pixel_values_size: torch.Size = None,
    quantize: bool = True,
    config: cfg.ModelConfig = None,
    lora_ranks: Optional[list[int]] = None,
    kv_cache_cls: type[kv_utils.KVCache] = kv_utils.KVCache,
    extra_exportable_prefill_kwargs: Union[
        dict[str, Any] | list[dict[str, Any]]
    ] = {},
    extra_exportable_decode_kwargs: dict[str, Any] = {},
    extra_unexportable_prefill_kwargs: Union[
        dict[str, Any] | list[dict[str, Any]]
    ] = {'skip_logits': True},
    extra_unexportable_decode_kwargs: dict[str, Any] = {},
):
  """Converts a nn.Module model to multi-signature tflite model.

  A PyTorch model will be converted to a tflite model with several signatures:
    * "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
        passed),
    * "prefill_[preill_seq_len]_pixel" (or "prefill_pixel" if only one
        prefill_seq_len is passed) if num_pixel_values > 0, and
    * "decode".

  "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
  passed) signature takes as a sample input:
    * a tensor of shape [1, prefill_seq_len] of token sequence,
    * a tensor of shape [1, prefill_seq_len] of token positions, and
    * an external KV cache.

  If num_pixel_values > 0, "prefill_[prefill_seq_len]_pixel" (or "prefill_pixel"
  if only one prefill_seq_len is passed) signature takes as a sample input:
    * a tensor of shape [1, prefill_seq_len] of token sequence,
    * a tensor of shape [1, prefill_seq_len] of token positions,
    * an external KV cache, and
    * a tensor of shape [1, num_pixel_values] of pixel values.

  "decode" signature takes as a sample input:
    * a tensor of shape [1, 1] of token sequence,
    * a tensor of shape [1, 1] of the token position, and
    * an external KV cache.

  The final tflite model will be exported to tflite_path.

  Args:
      pytorch_model (torch.nn.Module): PyTorch model to convert to tflite.
      output_path (str): The path to export the tflite model.
      output_name_prefix (str): The prefix of the tflite model name.
      prefill_seq_len (Union[int, list[int]]): The prefill sequence length to
        use. If a list, the model will have multiple prefill signatures.
      pixel_values_size (torch.Size, optional): The size of pixel values to pass
        to the model. If None, the model is not expected to take pixel values.
      quantize (bool, optional): Whether the model should be quanized. Defaults
        to True.
      config (cfg.ModelConfig, optional): The model config used to configure KV
        cache. If None, it uses the config of the pytorch_model.
      lora_ranks (list[int], optional): The ranks of the LORA layers. If None,
        no LoRA signatures will be added.
      kv_cache_cls (type[kv_utils.KVCache], optional): The KV cache class to
        use. Defaults to kv_utils.KVCache.
      extra_exportable_prefill_kwargs
        (dict[str, Any] | list[dict[str, Any]], optional): Extra arguments to
        pass to the prefill signatures. If a list, it should have the same
        length as prefill_seq_lens.
      extra_exportable_decode_kwargs (dict[str, Any], optional): Extra arguments
        to pass to the decode signature.
      extra_unexportable_prefill_kwargs
        (dict[str, Any] | list[dict[str, Any]], optional): Extra arguments to
        pass to the prefill model instead of the prefill signatures because they
        are not tensor-like arguments and not exportable. If a list, it should
        have the same length as prefill_seq_lens. Defaults to {'skip_logits':
        True}.
      extra_unexportable_decode_kwargs (dict[str, Any], optional): Extra
        arguments to pass to the decode model instead of the decode signature
        because they are not tensor-like arguments and not exportable.
  """
  # pylint: disable=protected-access
  torch._dynamo.config.cache_size_limit = 64

  config = config if config else pytorch_model.config
  prefill_seq_lens = (
      [prefill_seq_len] if isinstance(prefill_seq_len, int) else prefill_seq_len
  )
  loras = [None]
  if lora_ranks is not None:
    for rank in lora_ranks:
      lora = lora_utils.LoRA.zeros(rank, config)
      loras.append(lora)

  quant_suffix = 'q8' if quantize else 'f32'
  kv_size = config.kv_cache_max_len
  lora_suffix = (
      '' if not lora_ranks else f'_lora{",".join(map(str, lora_ranks))}'
  )
  output_filename = (
      f'{output_name_prefix}_{quant_suffix}_ekv{kv_size}{lora_suffix}.tflite'
  )
  output_file = os.path.join(output_path, output_filename)

  _export_helper(
      pytorch_model,
      output_file,
      prefill_seq_lens,
      pixel_values_size,
      quantize,
      config,
      loras,
      kv_cache_cls,
      extra_exportable_prefill_kwargs,
      extra_exportable_decode_kwargs,
      extra_unexportable_prefill_kwargs,
      extra_unexportable_decode_kwargs,
  )


def _export_helper(
    pytorch_model: torch.nn.Module,
    output_file: str,
    prefill_seq_lens: list[int],
    pixel_values_size: torch.Size,
    quantize: bool,
    config: cfg.ModelConfig,
    loras: list[None | lora_utils.LoRA],
    kv_cache_cls: type[kv_utils.KVCache],
    extra_exportable_prefill_kwargs: Union[
        dict[str, Any] | list[dict[str, Any]]
    ],
    extra_exportable_decode_kwargs: dict[str, Any],
    extra_unexportable_prefill_kwargs: Union[
        dict[str, Any] | list[dict[str, Any]]
    ],
    extra_unexportable_decode_kwargs: dict[str, Any],
):
  """Helper function to export a model to tflite."""
  prefill_model_list = []
  prefill_tokens_list = []
  prefill_input_pos_list = []
  for seq_len in prefill_seq_lens:
    unexportable_kwargs = (
        extra_unexportable_prefill_kwargs[i]
        if isinstance(extra_unexportable_prefill_kwargs, list)
        else extra_unexportable_prefill_kwargs
    )
    prefill_model_list.append(
        pytorch_model
        if unexportable_kwargs
        else ExportableModule(pytorch_model, **unexportable_kwargs)
    )
    prefill_tokens_list.append(torch.full((1, seq_len), 0, dtype=torch.int))
    prefill_input_pos_list.append(torch.arange(0, seq_len, dtype=torch.int))

  prefill_pixel_values = (
      torch.full(pixel_values_size, 0, dtype=torch.float32)
      if pixel_values_size
      else None
  )

  decode_model = (
      pytorch_model
      if not extra_unexportable_decode_kwargs
      else ExportableModule(pytorch_model, **extra_unexportable_decode_kwargs)
  )
  decode_token = torch.tensor([[0]], dtype=torch.int)
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  kv = kv_cache_cls.from_model_config(config)

  quant_config = quant_recipes.full_int8_dynamic_recipe() if quantize else None

  converter = converter_utils.Converter()
  for lora in loras:
    for i in range(len(prefill_seq_lens)):
      prefill_seq_len = prefill_seq_lens[i]
      prefill_tokens = prefill_tokens_list[i]
      prefill_input_pos = prefill_input_pos_list[i]
      prefill_signature_name = f'prefill_{prefill_seq_len}'

      sample_kwargs = {
          'tokens': prefill_tokens,
          'input_pos': prefill_input_pos,
          'kv_cache': kv,
          **(
              extra_exportable_prefill_kwargs[i]
              if isinstance(extra_exportable_prefill_kwargs, list)
              else extra_exportable_prefill_kwargs
          ),
      }

      if lora is not None:
        prefill_signature_name += f'_lora_r{lora.get_rank()}'
        sample_kwargs['lora'] = lora

      converter.add_signature(
          prefill_signature_name,
          prefill_model_list[i],
          sample_kwargs=sample_kwargs,
      )

      if prefill_pixel_values is not None:
        converter.add_signature(
            prefill_signature_name + '_pixel',
            prefill_model_list[i],
            sample_kwargs={
                **sample_kwargs,
                'pixel_values': prefill_pixel_values,
            },
        )

    sample_kwargs = {
        'tokens': decode_token,
        'input_pos': decode_input_pos,
        'kv_cache': kv,
        **extra_exportable_decode_kwargs,
    }
    if lora is not None:
      sample_kwargs['lora'] = lora

    converter.add_signature(
        'decode' if lora is None else f'decode_lora_r{lora.get_rank()}',
        decode_model,
        sample_kwargs=sample_kwargs,
    )

  edge_model = converter.convert(quant_config=quant_config)
  edge_model.export(output_file)
