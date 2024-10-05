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

"""A python script to build a SentencePieceModel protobuf from a HF tokenizer.

If a SentencePieceModel protobuf file is already available, it copies the
SentencePieceModel protobuf file instead of building a new one.

If not, it tries to build a SentencePieceModel protobuf file from the tokenizer
config files.
"""

import logging

from absl import app
from absl import flags
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as spm_model
import transformers

_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    None,
    "The path to the checkpoint where the tokenizer config files are.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "The path of the output SentencePieceModel protobuf file.",
)

_STRINGS_TO_VERIFY = flags.DEFINE_list(
    "strings_to_verify",
    [
        "Hello, world! How are you?",
        "Instruct: write a python program to add 2 and 3.",
    ],
    "The strings to verify the SentencePieceModel protobuf file.",
)


def _set_normalizer_spec(normalizer_spec: spm_model.NormalizerSpec):
  """Sets the normalizer spec compatible to BPE tokenizer."""
  normalizer_spec.add_dummy_prefix = False
  normalizer_spec.remove_extra_whitespaces = False
  normalizer_spec.escape_whitespaces = False


def _add_token(
    token: str,
    score: float,
    tokenizer: transformers.PreTrainedTokenizer,
    sp_model: spm_model.ModelProto,
    counts: dict[spm_model.ModelProto.SentencePiece.Type, int],
):
  """Adds a token to the SentencePieceModel protobuf with tis derived type."""
  unk_token = tokenizer.unk_token or tokenizer.pad_token or tokenizer.eos_token
  if token == unk_token:
    type = spm_model.ModelProto.SentencePiece.UNKNOWN
  elif token in tokenizer.special_tokens_map:
    type = spm_model.ModelProto.SentencePiece.CONTROL
    sp_model.trainer_spec.control_symbols.append(token)
  elif token in tokenizer.get_added_vocab():
    type = spm_model.ModelProto.SentencePiece.USER_DEFINED
    sp_model.trainer_spec.user_defined_symbols.append(token)
  else:
    type = spm_model.ModelProto.SentencePiece.NORMAL
  counts[type] = counts.get(type, 0) + 1

  # Replace 'Ġ' in BPE with a whitespace which SPM can handle properly. Without
  # this, the SPM tokenizer encodes " world" into " " and "world".
  normalized_token = token.replace("Ġ", " ")
  sp_model.pieces.add(piece=normalized_token, score=score, type=type)


def _build_spm_model_from_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
) -> spm_model.ModelProto:
  """Builds a SentencePieceModel protobuf from a tokenizer."""
  sp_model = spm_model.ModelProto()
  sp_model.trainer_spec.model_type = spm_model.TrainerSpec.BPE
  sp_model.trainer_spec.vocab_size = len(tokenizer.vocab)
  _set_normalizer_spec(sp_model.normalizer_spec)
  _set_normalizer_spec(sp_model.denormalizer_spec)

  id_to_token = {}
  for token, id in tokenizer.vocab.items():
    id_to_token[id] = token

  counts = {}
  for id in range(len(tokenizer.vocab)):
    _add_token(id_to_token[id], -id, tokenizer, sp_model, counts)

  logging.info("number of tokens: %d", len(sp_model.pieces))
  for type in counts:
    logging.info(
        "number of %s: %d",
        spm_model.ModelProto.SentencePiece.Type.Name(type),
        counts[type],
    )

  return sp_model


def main(_):
  tokenizer = transformers.AutoTokenizer.from_pretrained(_CHECKPOINT.value)
  if hasattr(tokenizer, "vocab_file"):
    logging.info("vocab_file exists: %s", tokenizer.vocab_file)
    with open(tokenizer.vocab_file, "rb") as f:
      sp_model = spm_model.ModelProto.FromString(f.read())
  else:
    logging.info("vocab_file does not exist. Try to build a new one.")
    sp_model = _build_spm_model_from_tokenizer(tokenizer)

  spm_serialized = sp_model.SerializeToString()

  # Verify the SentencePieceModel protobuf file.
  spm_tokenizer = spm.SentencePieceProcessor()
  spm_tokenizer.LoadFromSerializedProto(spm_serialized)
  for string in _STRINGS_TO_VERIFY.value:
    ids_by_tokenizer = tokenizer.encode(string)
    ids_by_spm = spm_tokenizer.encode(string)
    logging.info("String to verify: %s", string)
    logging.info("Token IDs by the oringal tokenizer: %s", ids_by_tokenizer)
    logging.info("Token IDs by the SentencePiece tokenizer: %s", ids_by_spm)
    # The original tokenizer may insert BOS token at the beginning.
    if ids_by_tokenizer == ids_by_spm or ids_by_tokenizer[1:] == ids_by_spm:
      logging.info("PASS")
    else:
      logging.warning("FAIL")

  logging.info(
      "Writing the SentencePieceModel protobuf file to: %s", _OUTPUT_PATH.value
  )
  with open(_OUTPUT_PATH.value, "wb") as f:
    f.write(spm_serialized)


if __name__ == "__main__":
  app.run(main)
