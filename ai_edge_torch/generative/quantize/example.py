# Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.
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

import numpy as np
import torch

import ai_edge_torch
from ai_edge_torch.generative.examples.gemma import gemma
from ai_edge_torch.generative.quantize import quant_recipes


def main():
  # Build a PyTorch model as usual
  config = gemma.get_fake_model_config_2b_for_test()
  model = gemma.Gemma(config)
  idx = torch.from_numpy(np.array([[1, 2, 3, 4]]))
  tokens = torch.full((1, 10), 0, dtype=torch.long, device="cpu")
  tokens[0, :4] = idx
  input_pos = torch.arange(0, 10)

  # Create a quantization recipe to be applied to the model
  quant_config = quant_recipes.full_linear_int8_dynamic_recipe()
  print(quant_config)

  # Convert with quantization
  edge_model = ai_edge_torch.convert(
      model, (tokens, input_pos), quant_config=quant_config
  )
  edge_model.export("/tmp/gemma_2b_quantized.tflite")


if __name__ == "__main__":
  main()
