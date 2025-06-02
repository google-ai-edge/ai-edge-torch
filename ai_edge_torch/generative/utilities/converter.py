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

import enum
import os
import pathlib
from typing import Optional, Union
from absl import flags
from ai_edge_torch._convert import converter as converter_utils
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import lora as lora_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.generative.utilities import export_config
from ai_edge_torch.quantize import quant_config as qcfg
import torch

ExportConfig = export_config.ExportConfig


class ExportableModule(torch.nn.Module):

  def __init__(self, module, **extra_kwargs):
    super().__init__()
    self.module = module
    self.extra_kwargs = extra_kwargs

  def forward(self, *export_args, **export_kwargs):
    full_kwargs = {**export_kwargs, **self.extra_kwargs}
    return self.module(*export_args, **full_kwargs)


class QuantizationName(str, enum.Enum):
  """Strings for all supported quantization recipes.

  none: No quantization.
  dynamic_int8: Dynamic range quantization with int8 weights.
  weight_only_int8: Weight only quantization with int8 weights.
  fp16: Float16 quantization.
  dynamic_int4_block32: Dynamic range quantization with int4 weights and block
  size of 32, better model quality but slower inference.
  dynamic_int4_block128: Dynamic range quantization with int4 weights and block
  size of 128, faster inference but worse model quality.
  """

  NONE = 'none'
  DYNAMIC_INT8 = 'dynamic_int8'
  WEIGHT_ONLY_INT8 = 'weight_only_int8'
  FP16 = 'fp16'
  DYNAMIC_INT4_BLOCK32 = 'dynamic_int4_block32'
  DYNAMIC_INT4_BLOCK128 = 'dynamic_int4_block128'


def define_conversion_flags(
    model_name: str,
    default_mask_as_input: bool = False,
    default_transpose_kv_cache: bool = False,
):
  """Defines common flags used for model conversion."""

  flags.DEFINE_string(
      'checkpoint_path',
      os.path.join(pathlib.Path.home(), f'Downloads/llm_data/{model_name}'),
      'The path to the model checkpoint, or directory holding the checkpoint.',
  )
  flags.DEFINE_string(
      'output_path',
      '/tmp/',
      'The path to export the tflite model.',
  )
  flags.DEFINE_string(
      'output_name_prefix',
      f'{model_name}',
      'The prefix of the output tflite model name.',
  )
  flags.DEFINE_multi_integer(
      'prefill_seq_lens',
      (8, 64, 128, 256, 512, 1024),
      'List of the maximum sizes of prefill input tensors.',
  )
  flags.DEFINE_integer(
      'kv_cache_max_len',
      1280,
      'The maximum size of KV cache buffer, including both prefill and decode.',
  )
  flags.DEFINE_string(
      'quantize',
      'dynamic_int8',
      'How the model should be quantized. Set to "none" to disable'
      ' quantization. See `QuantizationName` for supported quantization types.',
  )
  flags.DEFINE_multi_integer(
      'lora_ranks',
      None,
      'If set, the model will be converted with the provided list of LoRA'
      ' ranks.',
  )
  flags.DEFINE_bool(
      'mask_as_input',
      default_mask_as_input,
      'If true, the mask will be passed in as input. Otherwise, mask will be '
      'built by the model internally.',
  )
  flags.DEFINE_bool(
      'transpose_kv_cache',
      default_transpose_kv_cache,
      'If true, the model will be converted with transposed KV cache.',
  )
  flags.DEFINE_bool(
      'custom_checkpoint_loader',
      False,
      'If true, the conversion script will use a custom checkpoint loader which'
      ' will read a checkpoint from a remote source.',
  )
  return flags


def get_mask_cache_size_from_flags() -> int:
  """Returns the mask cache size according to the flags."""
  return 0 if flags.FLAGS.mask_as_input else flags.FLAGS.kv_cache_max_len


def get_quant_recipe_from_flag(
    quantize: str,
    model_config: cfg.ModelConfig,
) -> Optional[qcfg.QuantConfig]:
  """Processes the quantization flag and returns the corresponding recipe.

  Args:
      quantize: The quantization type.

  Returns:
      The quantization recipe, or None if no quantization is needed.

  Raises:
      ValueError: If the quantization type is not supported.
  """
  match quantize:
    case QuantizationName.NONE:
      return None
    case QuantizationName.DYNAMIC_INT8:
      return quant_recipes.full_int8_dynamic_recipe(mcfg=model_config)
    case QuantizationName.WEIGHT_ONLY_INT8:
      return quant_recipes.full_int8_weight_only_recipe(mcfg=model_config)
    case QuantizationName.FP16:
      return quant_recipes.full_fp16_recipe()
    case QuantizationName.DYNAMIC_INT4_BLOCK32:
      return quant_recipes.all_supported_int4_dynamic_block_recipe(
          32, mcfg=model_config
      )
    case QuantizationName.DYNAMIC_INT4_BLOCK128:
      return quant_recipes.all_supported_int4_dynamic_block_recipe(
          128, mcfg=model_config
      )
    case _:
      raise ValueError(f'Unsupported quantization flag: {quantize}')


def create_quantize_suffix(quantize: str) -> str:
  """Creates a suffix for the output file name based on the quantization type.

  Args:
      quantize: The quantization type.

  Returns:
      A string representing the quantization suffix.

  Raises:
      ValueError: If the quantization type is not supported.
  """
  match quantize:
    case QuantizationName.NONE:
      return 'f32'
    case QuantizationName.DYNAMIC_INT8:
      return 'q8'
    case QuantizationName.WEIGHT_ONLY_INT8:
      return 'q8_wo'
    case QuantizationName.FP16:
      return 'fp16'
    case QuantizationName.DYNAMIC_INT4_BLOCK32:
      return 'q4_block32'
    case QuantizationName.DYNAMIC_INT4_BLOCK128:
      return 'q4_block128'
    case _:
      raise ValueError(f'Unsupported quantization flag: {quantize}')


def _build_mask(mask_len, kv_cache_max_len, causal_mask_value) -> torch.Tensor:
  if isinstance(mask_len, list):
    return [
        _build_mask(i, kv_cache_max_len, causal_mask_value) for i in mask_len
    ]

  mask = torch.full(
      (mask_len, kv_cache_max_len), causal_mask_value, dtype=torch.float32
  )
  return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def convert_to_tflite(
    pytorch_model: torch.nn.Module,
    output_path: str,
    output_name_prefix: str,
    prefill_seq_len: Union[int, list[int]],
    kv_cache_max_len: int,
    pixel_values_size: torch.Size = None,
    pixel_seq_len: int = 0,
    quantize: str = 'dynamic_int8',
    config: cfg.ModelConfig = None,
    lora_ranks: Optional[list[int]] = None,
    export_config: ExportConfig = None,
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
      kv_cache_max_len (int): The maximum size of KV cache buffer, including
        both prefill and decode.
      pixel_values_size (torch.Size, optional): The size of pixel values to pass
        to the model. If None, the model is not expected to take pixel values.
      pixel_seq_len (int, optional): The length of pixel tokens, or pixel
        embeddings generated by the image encoder with pixel values. The actual
        length of prefill_seq_len will be added by pixel_seq_len when pixel
        values are passed.
      quantize (str, optional): The quantization type. Defaults to
        'dynamic_int8'.
      config (cfg.ModelConfig, optional): The model config used to configure KV
        cache. If None, it uses the config of the pytorch_model.
      lora_ranks (list[int], optional): The ranks of the LORA layers. If None,
        no LoRA signatures will be added.
      export_config (ExportConfig, optional): The export configuration. If None,
        it uses the default export configuration.
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

  quant_suffix = create_quantize_suffix(quantize)
  kv_size = kv_cache_max_len
  lora_suffix = (
      '' if not lora_ranks else f'_lora{",".join(map(str, lora_ranks))}'
  )

  if pixel_values_size is not None:
    assert pixel_seq_len > 0, 'pixel_seq_len must be greater than 0'
    max_prefill_seq_len = max(prefill_seq_lens)
    assert kv_size > max_prefill_seq_len + pixel_seq_len, (
        f'The KV cache size ({kv_size}) must be greater than the maximum '
        f'prefill sequence length ({max_prefill_seq_len}) + pixel sequence '
        f'length ({pixel_seq_len})'
    )

  if export_config is not None:
    if export_config.decode_batch_size > 1:
      output_name_prefix += f'_dbs{export_config.decode_batch_size}'

  output_filename = (
      f'{output_name_prefix}_{quant_suffix}_ekv{kv_size}{lora_suffix}.tflite'
  )
  output_file = os.path.join(output_path, output_filename)

  _export_helper(
      pytorch_model,
      output_file,
      prefill_seq_lens,
      kv_cache_max_len,
      pixel_values_size,
      pixel_seq_len,
      quantize,
      config,
      loras,
      export_config,
  )


def _export_helper(
    pytorch_model: torch.nn.Module,
    output_file: str,
    prefill_seq_lens: list[int],
    kv_cache_max_len: int,
    pixel_values_size: torch.Size,
    pixel_seq_len: int,
    quantize: str,
    config: cfg.ModelConfig,
    loras: list[None | lora_utils.LoRA],
    export_config: ExportConfig,
):
  """Helper function to export a model to tflite."""
  prefill_tokens_list = []
  prefill_input_pos_list = []
  for seq_len in prefill_seq_lens:
    prefill_tokens_list.append(torch.full((1, seq_len), 0, dtype=torch.int))
    prefill_input_pos_list.append(torch.arange(0, seq_len, dtype=torch.int))

  prefill_pixel_values = None
  prefill_tokens_list_with_pixel = []
  prefill_input_pos_list_with_pixel = []
  if pixel_values_size is not None:
    prefill_pixel_values = torch.full(pixel_values_size, 0, dtype=torch.float32)
    for seq_len in prefill_seq_lens:
      prefill_tokens_list_with_pixel.append(
          torch.full((1, seq_len + pixel_seq_len), 0, dtype=torch.int)
      )
      prefill_input_pos_list_with_pixel.append(
          torch.arange(0, seq_len + pixel_seq_len, dtype=torch.int)
      )

  prefill_masks = None
  if export_config.mask_as_input:
    prefill_masks = _build_mask(
        prefill_seq_lens, kv_cache_max_len, config.causal_mask_value
    )
    if not isinstance(prefill_masks, list):
      prefill_masks = [prefill_masks]
    assert len(prefill_masks) == len(prefill_seq_lens)

  decode_token = torch.tensor(
      [[0] for _ in range(export_config.decode_batch_size)], dtype=torch.int
  )
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  prefill_kv = kv_utils.KVCache.from_model_config(
      kv_cache_max_len, config, kv_layout=export_config.kvcache_layout
  )
  decode_kv = kv_utils.KVCache.from_model_config(
      kv_cache_max_len,
      config,
      batch_size=export_config.decode_batch_size,
      kv_layout=export_config.kvcache_layout,
  )

  quant_config = get_quant_recipe_from_flag(quantize, config)

  # For export, we create a module that captures any non-exportable,
  # arugments, e.g. the generation config object.
  mod = ExportableModule(pytorch_model, export_config=export_config).eval()

  converter = converter_utils.Converter()
  for lora in loras:
    for i in range(len(prefill_seq_lens)):
      prefill_seq_len = prefill_seq_lens[i]
      prefill_signature_name = f'prefill_{prefill_seq_len}'

      sample_kwargs = {
          'tokens': prefill_tokens_list[i],
          'input_pos': prefill_input_pos_list[i],
          'kv_cache': prefill_kv,
      }
      if prefill_masks is not None:
        sample_kwargs['mask'] = prefill_masks[i]

      if lora is not None:
        prefill_signature_name += f'_lora_r{lora.get_rank()}'
        sample_kwargs['lora'] = lora

      converter.add_signature(
          prefill_signature_name,
          mod,
          sample_kwargs=sample_kwargs,
      )

      if prefill_pixel_values is not None:
        sample_pixel_kwargs = {
            'tokens': prefill_tokens_list_with_pixel[i],
            'input_pos': prefill_input_pos_list_with_pixel[i],
            'kv_cache': prefill_kv,
            'pixel_values': prefill_pixel_values,
        }
        # mask should be built internally when pixel values are passed.
        if lora is not None:
          sample_pixel_kwargs['lora'] = lora
        converter.add_signature(
            prefill_signature_name + '_pixel',
            mod,
            sample_kwargs=sample_pixel_kwargs,
        )

    sample_kwargs = {
        'tokens': decode_token,
        'input_pos': decode_input_pos,
        'kv_cache': decode_kv,
    }
    if export_config.mask_as_input:
      # Note that the decode mask is not a correct causal mask, but it is okay
      # for the conversion purpose because only the shape matters in conversion.
      # A correct causal mask of decode for a given token position of decode, it
      # should be built like:
      #
      #  torch.triu(mask, diagonal=decode_position).unsqueeze(0).unsqueeze(0)
      #
      sample_kwargs['mask'] = _build_mask(
          1, kv_cache_max_len, config.causal_mask_value
      )
    if lora is not None:
      sample_kwargs['lora'] = lora

    converter.add_signature(
        'decode' if lora is None else f'decode_lora_r{lora.get_rank()}',
        mod,
        sample_kwargs=sample_kwargs,
    )

  edge_model = converter.convert(
      quant_config=quant_config,
  )
  edge_model.export(output_file)
