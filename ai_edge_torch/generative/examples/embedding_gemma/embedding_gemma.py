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

"""Embedding Gemma model."""

import os
from typing import Callable, Dict, Tuple, Union
import ai_edge_torch
from ai_edge_torch.generative.examples.embedding_gemma import heads
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import model_builder
import torch
from torch import nn

TENSOR_NAMES = loader.ModelLoader.TensorNames(
    ff_up_proj="layers.{}.mlp.up_proj",
    ff_down_proj="layers.{}.mlp.down_proj",
    ff_gate_proj="layers.{}.mlp.gate_proj",
    attn_query_proj="layers.{}.self_attn.q_proj",
    attn_key_proj="layers.{}.self_attn.k_proj",
    attn_value_proj="layers.{}.self_attn.v_proj",
    attn_output_proj="layers.{}.self_attn.o_proj",
    attn_query_norm="layers.{}.self_attn.q_norm",
    attn_key_norm="layers.{}.self_attn.k_norm",
    pre_attn_norm="layers.{}.input_layernorm",
    post_attn_norm="layers.{}.post_attention_layernorm",
    pre_ff_norm="layers.{}.pre_feedforward_layernorm",
    post_ff_norm="layers.{}.post_feedforward_layernorm",
    embedding="embed_tokens",
    final_norm="norm",
    lm_head=None,
)


def input_mask_to_positions(
    input_mask: torch.BoolTensor,  # Shape: (*B, T)
) -> torch.IntTensor:  # Shape: (*B, T)
  """Computes the `positions` from the `input_mask`.

  Args:
      input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
      The indices to use for RoPE and absolute position encodings for the given
      input mask.
  """
  positions = torch.cumsum(input_mask, dim=-1).to(torch.int32)
  return positions - (positions >= 1).to(torch.int32)


def sliding_window_int_to_pair(
    sliding_window_size: Union[int, Tuple[int, int]], is_causal: bool
) -> Tuple[int, int]:
  """Converts a sliding window size to a pair of left and right window sizes."""
  if isinstance(sliding_window_size, int):
    if is_causal:
      return (sliding_window_size, sliding_window_size)
    else:
      return (
          (sliding_window_size + 1) // 2,
          (sliding_window_size) // 2 + 1,
      )
  return sliding_window_size


def _make_sliding_window_mask(
    *,
    sliding_window_size: Union[int, Tuple[int, int]],
    q_pos: torch.IntTensor,  # Shape: (t, 1)
    kv_pos: torch.IntTensor,  # Shape: (1, s)
    is_causal: bool,
) -> torch.BoolTensor:  # Shape: (1, t, 1, s)
  """Makes a sliding window mask."""
  left_window_size, right_window_size = sliding_window_int_to_pair(
      sliding_window_size=sliding_window_size, is_causal=is_causal
  )

  dist = q_pos - kv_pos
  dist = dist.to(torch.float32)

  mask = torch.logical_or(
      (dist >= 0) & (dist < left_window_size),
      (dist < 0) & (-dist < right_window_size),
  )
  return mask[None, :, None, :]


def make_bidirectional_attn_mask(
    input_mask: torch.BoolTensor,  # Shape: (*B, L)
) -> torch.BoolTensor:  # Shape: (*B, L, 1, L)
  """Attention mask in batch mode.

  Args:
      input_mask: Input mask for the input

  Returns:
      Attention mask.
  """
  # (*B, L) -> (*B, L, 1) and (*B, 1, L)
  mask_1 = input_mask[..., None]
  mask_2 = input_mask[..., None, :]

  attn_mask = mask_1 & mask_2  # Logical AND (multiplication for bools)

  # (*B, L, L) -> (*B, L, 1, L)
  return attn_mask[..., :, None, :]


class EmbeddingGemma(nn.Module):
  """Embedding Gemma model."""

  def __init__(
      self,
      encoder: decoder.Decoder,
      pooling: nn.Module,
      projection: heads.ProjectionLayer,
      additional_projection: heads.ProjectionLayer | None = None,
      pad_id: int = 0,
      normalize_output: bool = True,
  ):
    super().__init__()
    self.encoder = encoder
    self.pooling = pooling
    self.projection = projection
    self.additional_projection = additional_projection
    self.normalize_output = normalize_output
    self.pad_id = pad_id
    self.encoder_config = encoder.config

  @torch.inference_mode
  def forward(
      self,
      text_batch: torch.Tensor,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    tokens = text_batch
    input_mask = tokens == self.pad_id
    # token embeddings of shape (b, t, n_embd)
    input_embeds = self.encoder.tok_embedding(tokens)
    if self.encoder_config.embedding_scale is not None:
      input_embeds = input_embeds * self.encoder_config.embedding_scale
    # RoPE parameters are the same for all blocks. Use the first layer.
    attn_config = self.encoder_config.block_config(0).attn_config

    input_pos = torch.arange(0, input_mask.shape[-1], dtype=torch.int32)
    # Different rotary base for global and local attention
    # based on attention pattern
    rope = [
        rotary_pos_emb.build_rope(
            input_pos,
            attn_config.head_dim,
            self.encoder_config.block_config(i).attn_config.rotary_base,
        )
        for i in range(self.encoder_config.num_layers)
    ]

    x = input_embeds
    encoder = self.encoder
    attn_mask = make_bidirectional_attn_mask(input_mask)

    q_len = input_embeds.shape[1]
    k_len = q_len
    sliding_window_size = encoder.config.block_config(
        0
    ).attn_config.sliding_window_size
    should_apply_sliding_window_mask = k_len > sliding_window_size
    if should_apply_sliding_window_mask:
      window_mask = _make_sliding_window_mask(
          sliding_window_size=sliding_window_size,
          q_pos=torch.arange(0, q_len, dtype=torch.int32)[..., None],
          kv_pos=torch.arange(0, k_len, dtype=torch.int32)[..., None, :],
          is_causal=False,
      )
      attn_mask_local = (attn_mask.float() * window_mask.float()).to(torch.bool)
    else:
      attn_mask_local = attn_mask.clone()

    attn_mask = attn_mask.permute(0, 2, 1, 3)
    attn_mask_local = attn_mask_local.permute(0, 2, 1, 3)

    for i, block in enumerate(self.encoder.transformer_blocks):
      is_global = self.encoder_config.check_if_global_attention_layer(i)
      mask_entry = attn_mask if is_global else attn_mask_local
      kv_entry = None
      x, _ = block(x, rope[i], mask_entry, input_pos, kv_entry)
    x = self.encoder.final_norm(x)

    pooled_x = self.pooling(x, input_mask)
    x = self.projection(pooled_x)
    if self.additional_projection is not None:
      x = self.additional_projection(x)
    if self.normalize_output:
      x = torch.nn.functional.normalize(x, p=2, dim=1)
    return {"encodings": x}


def get_encoder_config() -> cfg.ModelConfig:
  """Returns the encoder transformer config."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=1152,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=3,
        head_dim=256,
        num_query_groups=1,
        rotary_base=1_000_000 if (idx + 1) % 6 == 0 else 10_000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=512,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if (idx + 1) % 6 == 0
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  num_layers = 24
  embedding_dim = 768
  config = cfg.ModelConfig(
      vocab_size=262_144,
      num_layers=num_layers,
      max_seq_len=2048,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=None,
      enable_hlfb=False,
  )
  return config


def build_embedding_gemma(
    checkpoint_path: str,
    normalize_output: bool = True,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  """Builds an Embedding Gemma 300M model."""
  # TODO(b/403644647): Better error handling for loading checkpoints with
  # different tensor names.
  encoder_config = get_encoder_config()
  encoder = model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=encoder_config,
      tensor_names=TENSOR_NAMES,
      model_class=decoder.Decoder,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )
  assert encoder is not None, "Failed to load encoder from checkpoint."
  assert encoder_config is not None, "Failed to load encoder config."
  pooling_layer = heads.MeanPooling()
  linear_1 = heads.ProjectionLayer(
      in_features=encoder_config.embedding_dim,
      out_features=3072,
  )
  linear_1_tensors = loader.load_safetensors(
      os.path.join(checkpoint_path, "2_Dense")
  )
  linear_1.load_state_dict(linear_1_tensors, strict=False)
  linear_2 = heads.ProjectionLayer(
      in_features=3072,
      out_features=encoder_config.embedding_dim,
  )
  linear_2_tensors = loader.load_safetensors(
      os.path.join(checkpoint_path, "3_Dense")
  )
  linear_2.load_state_dict(linear_2_tensors, strict=False)
  model = EmbeddingGemma(
      encoder=encoder,
      pooling=pooling_layer,
      projection=linear_1,
      additional_projection=linear_2,
      normalize_output=normalize_output,
  )
  model.eval()
  return model


def convert_to_litert(
    pytorch_model: EmbeddingGemma,
    output_path: str,
    output_name_prefix: str,
    prefill_seq_len: int,
    quantize: str = "dynamic_int8",
    **kwargs,
):
  """Converts a PyTorch model to LITert."""
  del kwargs
  seq_len = prefill_seq_len
  quant_suffix = converter.create_quantize_suffix(quantize)
  output_filename = f"{output_name_prefix}_{quant_suffix}.tflite"
  output_file = os.path.join(output_path, output_filename)
  token_batch_input = torch.full((1, seq_len), 0, dtype=torch.int)
  quant_config = converter.get_quant_recipe_from_flag(
      quantize, pytorch_model.encoder.config
  )

  cvt = ai_edge_torch.signature(
      f"embed_{seq_len}",
      pytorch_model,
      sample_kwargs={"text_batch": token_batch_input},
  ).convert(quant_config=quant_config)
  cvt.export(output_file)
