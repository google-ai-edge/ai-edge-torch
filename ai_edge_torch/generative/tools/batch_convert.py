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

"""A python script to convert a batch of Generative models to TF Lite."""

import dataclasses
import enum
import logging
import os
import pathlib
from typing import Callable, Sequence

from absl import app
from absl import flags
from ai_edge_torch.generative.examples.gemma import gemma1
from ai_edge_torch.generative.examples.gemma import gemma2
from ai_edge_torch.generative.examples.llama import llama
from ai_edge_torch.generative.examples.openelm import openelm
from ai_edge_torch.generative.examples.phi import phi2
from ai_edge_torch.generative.examples.phi import phi3
from ai_edge_torch.generative.examples.smollm import smollm
from ai_edge_torch.generative.examples.tiny_llama import tiny_llama
from ai_edge_torch.generative.utilities import converter
import torch

_CHECKPOINT_ROOT_PATH = flags.DEFINE_string(
    "checkpoint_root_path",
    os.path.join(pathlib.Path.home(), "Downloads/llm_data/"),
    "The root path to the checkpoints.",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    os.path.join(pathlib.Path.home(), "models"),
    "The output directory to store the converted models.",
)


@enum.unique
class ExportPrecision(enum.Enum):
  """Specifies the precision of the exported model."""

  INT8 = enum.auto()
  FP32 = enum.auto()


@dataclasses.dataclass
class ConversionConfig:
  """A dataclass to store the conversion config for a model."""

  model_name: str
  input_checkpoint: str
  tflite_output_path: str
  prefill_seq_len: int
  kv_cache_max_len: int
  export_precision: Sequence[ExportPrecision]
  model_builder: Callable[..., torch.nn.Module]

  def print_config(self) -> None:
    """Prints the conversion config."""
    logging.info("Model name: %s", self.model_name)
    logging.info("Input checkpoint: %s", self.input_checkpoint)
    logging.info("TF Lite output path: %s", self.tflite_output_path)
    logging.info("Prefill seq len: %s", self.prefill_seq_len)
    logging.info("KV cache max len: %s", self.kv_cache_max_len)
    logging.info("Export precision: %s", self.export_precision)


def prepare_conversion_configs() -> Sequence[ConversionConfig]:
  """Prepares the conversion configs for a batch of models."""
  conversion_configs = [
      ConversionConfig(
          model_name="tinyllama",
          input_checkpoint=os.path.join(
              _CHECKPOINT_ROOT_PATH.value, "tiny_llama"
          ),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "tiny_llama"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=tiny_llama.build_model,
      ),
      ConversionConfig(
          model_name="gemma",
          input_checkpoint=os.path.join(
              _CHECKPOINT_ROOT_PATH.value, "gemma-2b"
          ),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "gemma"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=gemma1.build_2b_model,
      ),
      ConversionConfig(
          model_name="gemma2",
          input_checkpoint=os.path.join(
              _CHECKPOINT_ROOT_PATH.value, "gemma2-2b"
          ),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "gemma2"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=gemma2.build_2b_model,
      ),
      ConversionConfig(
          model_name="llama",
          input_checkpoint=os.path.join(_CHECKPOINT_ROOT_PATH.value, "llama"),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "llama"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=llama.build_model,
      ),
      ConversionConfig(
          model_name="phi2",
          input_checkpoint=os.path.join(_CHECKPOINT_ROOT_PATH.value, "phi2"),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "phi2"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=phi2.build_model,
      ),
      ConversionConfig(
          model_name="phi3",
          input_checkpoint=os.path.join(_CHECKPOINT_ROOT_PATH.value, "phi3"),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "phi3"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=phi3.build_model,
      ),
      ConversionConfig(
          model_name="openelm",
          input_checkpoint=os.path.join(_CHECKPOINT_ROOT_PATH.value, "openelm"),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "openelm"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=openelm.build_model,
      ),
      ConversionConfig(
          model_name="smollm",
          input_checkpoint=os.path.join(_CHECKPOINT_ROOT_PATH.value, "smollm"),
          tflite_output_path=os.path.join(_OUTPUT_DIR.value, "smollm"),
          prefill_seq_len=1024,
          kv_cache_max_len=1280,
          export_precision=[ExportPrecision.INT8, ExportPrecision.FP32],
          model_builder=smollm.build_model,
      ),
  ]
  return conversion_configs


def get_output_filename(
    model_name: str,
    precision: ExportPrecision,
    prefill_seq_len: int,
    kv_cache_max_len: int,
) -> str:
  """Returns the output filename for a converted TF Litemodel."""
  if precision == ExportPrecision.INT8:
    precision_str = "q8"
  elif precision == ExportPrecision.FP32:
    precision_str = "f32"
  else:
    raise ValueError(f"Unsupported precision: {precision}")
  return f"{model_name}_{precision_str}_seq{prefill_seq_len}_ekv{kv_cache_max_len}.tflite"


def convert_models(conversion_configs: Sequence[ConversionConfig]) -> None:
  """Executes the conversion for a batch of models specified by the `conversion_configs`."""
  for config in conversion_configs:
    logging.info(
        "Converting model: %s with the following config:", config.model_name
    )
    config.print_config()
    pytorch_model = config.model_builder(
        config.input_checkpoint, kv_cache_max_len=config.kv_cache_max_len
    )
    for precision in config.export_precision:
      output_filename = get_output_filename(
          config.model_name,
          precision,
          config.prefill_seq_len,
          config.kv_cache_max_len,
      )
      converter.convert_to_tflite(
          pytorch_model,
          tflite_path=os.path.join(config.tflite_output_path, output_filename),
          prefill_seq_len=config.prefill_seq_len,
          quantize=True if precision == ExportPrecision.INT8 else False,
      )
      logging.info("Successfully converted model: %s", output_filename)


def main(_):
  convert_models(prepare_conversion_configs())


if __name__ == "__main__":
  app.run(main)
