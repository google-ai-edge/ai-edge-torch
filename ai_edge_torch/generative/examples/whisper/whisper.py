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



class EdgeTorchWhisperEncoder(nn.Module):
    def __init__(self,
                 cfg: cfg.AudioEncoderConfig
                 enable_hlfb = True):

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.head_dim = embed_dim // num_heads

        pre_attention_norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
        )

        attn_config = cfg.AttentionConfig(
            num_heads = self.num_heads,
            head_dim = self.head_dim,
            num_query_groups = self.num_heads,
            enable_kv_cache = False,
            qkv_use_bias = True,
            qkv_transpose_before_split = True,
            output_proj_use_bias = True,
        )

        pre_ff_norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
        )

        ff_config = cfg.FeedForwardConfig(
            type=cfg.FeedForwardType.SEQUENTIAL,
            activation=cfg.ActivationConfig(cfg.ActivationType.GELU),
            intermediate_size=self.encoder_ffn_dim,
            use_bias = True,
            pre_ff_norm_config=pre_ff_norm_config
        )

        self.block_config = cfg.TransformerBlockConfig(
            attn_config=attn_config,
            ff_config=ff_config,
            pre_attention_norm_config=pre_attention_norm_config,
        )

        torch_encoder_config = encoder_config

        self.final_norm_config = cfg.NormalizationConfig(
            type=cfg.NormalizationType.LAYER_NORM,
        )
        self.model_config = cfg.ModelConfig(
            vocab_size = torch_encoder_config.vocab_size,
            num_layers = torch_encoder_config.num_hidden_layers,
            max_seq_len = torch_encoder_config.max_target_positions,
            embedding_dim = torch_encoder_config.d_model,
            block_configs = (self.block_config,),
            enable_hlfb = enable_hlfb,
            final_norm_config=self.final_norm_config
        )

        self.embed_positions = nn.Embedding(
            torch_encoder_config.max_source_positions, self.model_config.embedding_dim
        )

        self.conv1 = nn.Conv1d(80, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.transformer_blocks = nn.ModuleList(
            aie_attention.TransformerBlock(self.model_config.block_configs[0], self.model_config)
                for _ in range(self.model_config.num_layers)
        )

        self.layer_norm = builder.build_norm(embed_dim, self.model_config.final_norm_config)
        self.k_cross_modules = nn.ModuleList(nn.Linear(embed_dim, embed_dim, bias = False) for _ in range(self.model_config.num_layers))
        self.v_cross_modules = nn.ModuleList(nn.Linear(embed_dim, embed_dim, bias = True) for _ in range(self.model_config.num_layers))

        self.k_cross_cache = None
        self.v_cross_cache = None

        self.pre_transpose_k_cross = pre_transpose_k_cross

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(self, input_features):

        x = nn.functional.gelu(self.conv1(input_features))
        x = nn.functional.gelu(self.conv2(x))

        input_embeds = x.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight

        x = input_embeds + embed_pos

        full_self_attention_mask = torch.zeros(1, 1, 1500, 1500)

        for _, block in enumerate(self.transformer_blocks):

            x = block(x, mask = full_self_attention_mask)

        encoder_latents = self.layer_norm(x)

        for i in range(len(self.k_cross_modules)):
            if self.k_cross_cache is None:
                self.k_cross_cache = self._shape(self.k_cross_modules[i](encoder_latents), -1, 1)
                self.v_cross_cache = self._shape(self.v_cross_modules[i](encoder_latents), -1, 1)
            else:
                self.k_cross_cache = torch.cat((self.k_cross_cache, self._shape(self.k_cross_modules[i](encoder_latents), -1, 1)), dim = 0)
                self.v_cross_cache = torch.cat((self.v_cross_cache, self._shape(self.v_cross_modules[i](encoder_latents), -1, 1)), dim = 0)

        return self.k_cross_cache, self.v_cross_cache


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
