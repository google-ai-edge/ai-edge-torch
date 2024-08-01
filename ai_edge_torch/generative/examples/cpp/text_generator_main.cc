/* Copyright 2024 The AI Edge Torch Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

// This is a simplified example of using TFLite to generate text.
// Please note that this is only a starting point and the user is expected
// to create their own pipeline potentially using different tokenizers and
// better samplers.
//
// Example usage:
//  generate_main --tflite_model="PATH/model.tflite" \
//    --sentencepiece_model="PATH/sp.model" \
//    --prompt="Write an email:" \
//    --max_decode_steps=64 \
//    --start_token="<bos>" \
//    --stop_token="<eos>"  \
//    --num_threads=4 \

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model prepared for text generation using ODML "
          "tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to sentencepiece model.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt to the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "The number of tokens to generate. Defaults to maximum Sequence size "
          "defined during conversion.");
ABSL_FLAG(std::string, start_token, "",
          "Start token is appended to the beginning of input prompt to "
          "signify start of sentence.");
ABSL_FLAG(std::string, stop_token, "",
          "Stop token used to deterine end of decoding loop. If not provided "
          "will decode until max_Seq_len or max_decode_steps.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");

namespace {

// Prepare helpers
std::unique_ptr<tflite::FlatBufferModel> LoadModel() {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          absl::GetFlag(FLAGS_tflite_model).c_str());
  TFLITE_MINIMAL_CHECK(model != nullptr);
  return model;
}

std::unique_ptr<tflite::Interpreter> BuildInterpreter(
    tflite::FlatBufferModel* model, int num_threads) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // NOTE: We need to manually register optimized OPs for KV-cache and
  // Scaled Dot Product Attention (SDPA).
  tflite::ops::custom::GenAIOpsRegisterer(&resolver);
  tflite::InterpreterBuilder builder(*model, resolver);
  TFLITE_MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);
  return interpreter;
}

std::unique_ptr<sentencepiece::SentencePieceProcessor>
LoadSentencePieceProcessor() {
  std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model),
                      std::ios::binary);
  std::string serialized_proto = std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  TFLITE_MINIMAL_CHECK(
      processor->LoadFromSerializedProto(serialized_proto).ok());
  return processor;
}

// A basic greedy sampler (equivalent to argmax).
int GreedySampler(const TfLiteTensor* logits) {
  float max_value = -std::numeric_limits<float>::infinity();
  int max_index = 0;
  // logits shape: [Batch, Seq, Vocab], Dtype: float
  for (int i = 0; i < logits->dims->data[2]; ++i) {
    if (logits->data.f[i] > max_value) {
      max_value = logits->data.f[i];
      max_index = i;
    }
  }
  return max_index;
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  // Prepare required components.
  std::unique_ptr<tflite::FlatBufferModel> model = LoadModel();
  std::unique_ptr<tflite::Interpreter> interpreter =
      BuildInterpreter(model.get(), absl::GetFlag(FLAGS_num_threads));
  std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_processor =
      LoadSentencePieceProcessor();

  // Get prefill and decode signature runners and allocate tensors per
  // signature.
  auto prefill_runner = interpreter->GetSignatureRunner("prefill");
  TFLITE_MINIMAL_CHECK(prefill_runner->AllocateTensors() == kTfLiteOk);
  auto decode_runner = interpreter->GetSignatureRunner("decode");
  TFLITE_MINIMAL_CHECK(decode_runner->AllocateTensors() == kTfLiteOk);

  // Get Input Tensors for each of the runners.
  // Shape: [Batch, Seq], Dtype: int64
  TfLiteTensor* prefill_input = prefill_runner->input_tensor("args_0");
  // Shape: [Seq], Dtype: int64
  TfLiteTensor* prefill_input_pos = prefill_runner->input_tensor("args_1");
  // Shape: [Batch, Seq], Dtype: int64
  TfLiteTensor* decode_input = decode_runner->input_tensor("args_0");
  // Shape: [Seq], Dtype: int64
  TfLiteTensor* decode_input_pos = decode_runner->input_tensor("args_1");
  int max_seq_size = prefill_input->dims->data[1];

  // Tokenize the input prompt.
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  std::vector<int> prompt_tokens;
  TFLITE_MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

  std::string start_token = absl::GetFlag(FLAGS_start_token);
  if (!start_token.empty()) {
    prompt_tokens.insert(prompt_tokens.begin(),
                         sp_processor->PieceToId((start_token)));
  }
  std::string stop_token = absl::GetFlag(FLAGS_stop_token);
  int stop_token_id = -1;
  if (!stop_token.empty()) {
    stop_token_id = sp_processor->PieceToId((stop_token));
  }

  // Fill in the inputs (assuming one batch).
  // NOTE: We skip the last token and use that during decode.
  int prefill_seq_size =
      std::min(static_cast<int>(prompt_tokens.size()), max_seq_size);
  for (int i = 0; i < prefill_seq_size - 1; ++i) {
    prefill_input->data.i64[i] = prompt_tokens[i];
    prefill_input_pos->data.i64[i] = i;
  }
  TFLITE_MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);

  // Decode until max sequence size or user defined step limit, whichever is
  // smaller.
  // NOTE: max kv-cache size is *not* necessarily the same size as the max
  // sequence length. KV Cache buffer wraps around if exahusted before max
  // sequence length or stopping criteria reach.
  int max_decode_steps = absl::GetFlag(FLAGS_max_decode_steps) == -1
                             ? max_seq_size
                             : absl::GetFlag(FLAGS_max_decode_steps);
  int decode_steps =
      std::min(max_decode_steps, max_seq_size - prefill_seq_size);
  TFLITE_MINIMAL_CHECK(decode_steps > 0);

  std::vector<int> output_tokens;
  output_tokens.reserve(decode_steps);
  int next_token = prompt_tokens[prefill_seq_size - 1];
  int next_position = prefill_seq_size - 1;
  for (int i = 0; i < decode_steps; ++i) {
    decode_input->data.i64[0] = next_token;
    decode_input_pos->data.i64[0] = next_position;
    TFLITE_MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
    next_token = GreedySampler(decode_runner->output_tensor("output_0"));
    output_tokens.push_back(next_token);
    next_position += 1;
    if (next_token == stop_token_id) {
      break;
    }
  }

  // Detokenize the generated output.
  std::string output_text;
  TFLITE_MINIMAL_CHECK(sp_processor->Decode(output_tokens, &output_text).ok());

  printf("Prompt:\n%s\nOutput text:\n%s\n", prompt.c_str(),
         output_text.c_str());

  return 0;
}
