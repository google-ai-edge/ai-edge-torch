"""Example of building the Whisper model."""

import os
import pathlib
from typing import Optional, Tuple
from absl import app

from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.layers.normalization as normalization
import ai_edge_torch.generative.utilities.whisper_loader as loading_utils
import torch
from torch import nn
import torch.nn as nn



TENSOR_NAMES = loader.WhisperEncoderModelLoader.TensorNames(
    attn_key_proj = "layers.{}.self_attn.k_proj",
    attn_value_proj = "layers.{}.self_attn.v_proj",
    attn_query_proj = "layers.{}.self_attn.q_proj",
    attn_output_proj = "layers.{}.self_attn.out_proj",
    pre_attn_norm = "layers.{}.self_attn_layer_norm",
    pre_ff_norm = "layers.{}.final_layer_norm",
    ff_up_proj = "layers.{}.fc1",
    ff_down_proj = "layers.{}.fc2",
    conv1D_0 = "conv1",
    conv1D_1 = "conv2",
    layer_norm = "layer_norm",
    embed_positions = "embed_positions"
)


class WhisperEncoderAiEdgeTorch(nn.Module):
    def __init__(self, config: cfg.ModelConfig):
        super().__init__()
        self.config = config

        transformer_config = config.block_configs[0]

        NUM_MEL_BINS = 80
        self.conv1 = nn.Conv1d(
            in_channels=NUM_MEL_BINS,
            out_channels=config.embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.embedding_dim,
            out_channels=config.embedding_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.embed_positions = nn.Embedding(
            config.max_seq_len, config.embedding_dim
        )

        self.transformer_blocks = nn.ModuleList(
            attention.TransformerBlock(config.block_config(0), config)
            for idx in range(config.num_layers)
        )

        self.layer_norm = builder.build_norm(config.embedding_dim, config.final_norm_config)

    @torch.inference_mode
    def forward(self, input_features):
        x = nn.functional.gelu(self.conv1(input_features))
        x = nn.functional.gelu(self.conv2(x))

        inputs_embeds = x.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight
        x = inputs_embeds + embed_pos

        full_self_attention_mask = torch.zeros(1, 1, 1500, 1500)
        for _, block in enumerate(self.transformer_blocks):
            x = block(x, mask = full_self_attention_mask)

        return self.layer_norm(x)


def get_model_config(checkpoint_path: str) -> cfg.ModelConfig:
    torch_model = torch.load(checkpoint_path)
    torch_encoder_config = torch_model.model.get_encoder().state_dict()["config"]

    pre_attention_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    attention_config = cfg.AttentionConfig(
        num_heads = torch_encoder_config.num_attention_heads,
        head_dim = torch_encoder_config.hidden_size // torch_encoder_config.num_attention_heads,
        num_query_groups = torch_encoder_config.num_attention_heads,
        enable_kv_cache = False,
        qkv_use_bias = True,
        qkv_transpose_before_split = True,
        output_proj_use_bias = True
    )

    pre_ff_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    feedforward_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.SEQUENTIAL,
        activation=cfg.ActivationConfig(cfg.ActivationType.GELU),
        intermediate_size=torch_encoder_config.encoder_ffn_dim,
        use_bias = True,
        pre_ff_norm_config = pre_ff_norm_config
    )

    encoder_block = cfg.TransformerBlockConfig(
        attn_config = attention_config,
        ff_config = feedforward_config,
        pre_attention_norm_config = pre_attention_norm_config
    )

    final_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    whisperModelConfig = cfg.ModelConfig(
        vocab_size = torch_encoder_config.vocab_size,
        num_layers = torch_encoder_config.num_hidden_layers,
        max_seq_len = torch_encoder_config.max_source_positions,
        embedding_dim = torch_encoder_config.d_model,
        block_configs = (encoder_block,),
        enable_hlfb = False,
        final_norm_config = final_norm_config,
    )
    return whisperModelConfig

def build_encoder(checkpoint_path: str, **kwargs) -> nn.Module:
  config = get_model_config(checkpoint_path)
  encoder = WhisperEncoderAIEdgeTorch(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(encoder, strict=True)
  return encoder 

def main(_):
  # 
  HF_PATH = os.path.join(pathlib.Path.home(), "Downloads/llm_data/whisper-tiny")

  test_data_path = pathlib.Path(__file__).parent.resolve()

  encoder = build_encoder(HF_PATH)
  encoder.eval()


if __name__ == "__main__":
  app.run(main)
