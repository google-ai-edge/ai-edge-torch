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
"""Verification utilities for EmbeddingGemma-300M."""

from ai_edge_torch.generative.examples.embedding_gemma import embedding_gemma
from huggingface_hub import snapshot_download  # pylint: disable=g-importing-member
from sentence_transformers import SentenceTransformer
import torch
import transformers


DEFAULT_PROMPTS = [
    "What is the meaning of life?",
    "This is an example sentence.",
]


def _mean_pool(last_hidden_states, attention_mask):
  """Mean pooling of hidden states, ignoring padding tokens."""
  masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
  sum_hidden_states = masked_hidden_states.sum(dim=1)
  count = attention_mask.sum(dim=1).unsqueeze(-1)
  count = torch.clamp(count, min=1e-9)
  return sum_hidden_states / count


def verify_embedding_gemma_300m(
    checkpoint_dir: str,
    prompts: list[str] | None = None,
    atol: float = 0.25,
) -> bool:
  """Verifies EmbeddingGemma-300M."""
  try:
    print(f"Downloading model from: {checkpoint_dir}")
    model_path = snapshot_download(repo_id=checkpoint_dir)
    print(f"Model downloaded to: {model_path}")
  except Exception as e:
    print(f"Error downloading model {checkpoint_dir}: {e}")
    return False

  print(f"Loading tokenizer from: {model_path}")
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

  print(f"Loading original model from: {model_path}")
  try:
    original_model = SentenceTransformer(model_path)
    original_model.eval()
  except Exception as e:
    print(f"Failed to load original model: {e}")
    return False

  print(f"Building reauthored model from: {model_path}")
  try:
    reauthored_model = embedding_gemma.build_model(model_path)
    reauthored_model.eval()
  except (OSError, ValueError) as e:
    print(f"Failed to build or load reauthored model: {e}")
    return False

  prompts_to_run = prompts if prompts is not None else DEFAULT_PROMPTS
  print(f"Tokenizing prompts: {prompts_to_run}")
  inputs = tokenizer(
      prompts_to_run, return_tensors="pt", padding=True, truncation=True
  )
  tokens, attention_mask = inputs["input_ids"], inputs["attention_mask"]

  print("Running inference...")
  with torch.no_grad():
    # SentenceTransformer model directly returns embeddings
    original_embeddings = original_model.encode(
        prompts_to_run, convert_to_tensor=True
    )
    print(f"Original embeddings shape: {original_embeddings.shape}")
    # Reauthored model includes pooling and norm in forward pass
    reauthored_embeddings = reauthored_model(
        tokens, attention_mask=attention_mask
    )

  if not torch.allclose(original_embeddings, reauthored_embeddings, atol=atol):
    print("Verification failed: Embeddings do not match!")
    print(
        "Max difference:"
        f" {torch.max(torch.abs(original_embeddings - reauthored_embeddings))}"
    )
    return False

  print("Verification successful: Embeddings match.")
  return True
