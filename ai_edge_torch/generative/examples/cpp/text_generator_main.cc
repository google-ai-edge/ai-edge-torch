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
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/match.h"
#include "ai_edge_torch/generative/examples/cpp/utils.h"
#include "src/sentencepiece_processor.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/experimental/genai/genai_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/signature_runner.h"

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
//    --num_threads=4

ABSL_FLAG(std::string, tflite_model, "",
          "Two-signature tflite model prepared for text generation using ODML "
          "tools.");
ABSL_FLAG(std::string, sentencepiece_model, "", "Path to sentencepiece model.");
ABSL_FLAG(std::string, prompt, "Write an email:", "Input prompt to the model.");
ABSL_FLAG(int, max_decode_steps, -1,
          "The number of tokens to generate. Defaults to the KV cache size "
          "defined during conversion.");
ABSL_FLAG(std::string, start_token, "",
          "Start token is appended to the beginning of input prompt to "
          "signify start of sentence.");
ABSL_FLAG(std::string, stop_token, "",
          "Stop token used to deterine end of decoding loop. If not provided "
          "will decode until max_kv_cache_size or max_decode_steps.");
ABSL_FLAG(int, num_threads, 4, "Number of threads to use. Defaults to 4.");
ABSL_FLAG(std::string, weight_cache_path, "",
          "XNNPACK weight caching path, e.g. /tmp/model.xnnpack_cache.");

namespace {

using ai_edge_torch::examples::AlignedAllocator;

std::unique_ptr<tflite::FlatBufferModel> LoadModel() {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          absl::GetFlag(FLAGS_tflite_model).c_str());
  MINIMAL_CHECK(model != nullptr);
  return model;
}

void ApplyXNNPACKWeightCaching(tflite::Interpreter* interpreter) {
  auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
  std::string weight_cache_path = absl::GetFlag(FLAGS_weight_cache_path);
  delegate_options.weight_cache_file_path = weight_cache_path.c_str();
  delegate_options.num_threads = absl::GetFlag(FLAGS_num_threads);
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING;
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;

  MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(
                    tflite::Interpreter::TfLiteDelegatePtr(
                        TfLiteXNNPackDelegateCreate(&delegate_options),
                        [](TfLiteDelegate* delegate) {
                          TfLiteXNNPackDelegateDelete(delegate);
                        })) == kTfLiteOk);
}

std::unique_ptr<tflite::Interpreter> BuildInterpreter(
    tflite::FlatBufferModel* model, int num_threads) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // NOTE: We need to manually register optimized OPs for KV-cache and
  // Scaled Dot Product Attention (SDPA).
  tflite::ops::custom::GenAIOpsRegisterer(&resolver);
  tflite::InterpreterBuilder builder(*model, resolver);
  MINIMAL_CHECK(builder.SetNumThreads(num_threads) == kTfLiteOk);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  MINIMAL_CHECK(interpreter != nullptr);

  if (!absl::GetFlag(FLAGS_weight_cache_path).empty()) {
    // optionally use xnnpack with weight caching
    ApplyXNNPACKWeightCaching(interpreter.get());
  }

  return interpreter;
}

std::map<std::string, std::vector<float, AlignedAllocator<float>>> BuildKVCache(
    tflite::Interpreter* interpreter) {
  tflite::SignatureRunner* runner = interpreter->GetSignatureRunner("decode");
  if (runner == nullptr) {
    return {};
  }
  // The two arguments excluded are `tokens` and `input_pos`.
  size_t num_layers = (runner->input_size() - 2) / 2;
  if (num_layers == 0) {
    return {};
  }

  std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache;
  for (int i = 0; i < num_layers; ++i) {
    std::string k_cache_name = "kv_cache_k_" + std::to_string(i);
    std::string v_cache_name = "kv_cache_v_" + std::to_string(i);
    // We are assuming K and V tensors are of the same shape.
    TfLiteTensor* tensor = runner->input_tensor(k_cache_name.c_str());
    size_t count = tensor->bytes / sizeof(float);
    kv_cache.emplace(k_cache_name,
                     std::vector<float, AlignedAllocator<float>>(count, 0.0));
    kv_cache.emplace(v_cache_name,
                     std::vector<float, AlignedAllocator<float>>(count, 0.0));
  }

  return kv_cache;
}

void PrepareRunner(
    tflite::SignatureRunner* runner,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache) {
  for (auto& [name, cache] : kv_cache) {
    TfLiteCustomAllocation allocation = {
        .data = static_cast<void*>(cache.data()),
        .bytes = cache.size() * sizeof(float)};
    // Both input and output tensors are set to the same buffer. Not all
    // delegates support this in-place update. For those cases, we need to do
    // a ping-pong buffer and update the pointers between inference calls.
    MINIMAL_CHECK(runner->SetCustomAllocationForInputTensor(
                      name.c_str(), allocation) == kTfLiteOk);
    MINIMAL_CHECK(runner->SetCustomAllocationForOutputTensor(
                      name.c_str(), allocation) == kTfLiteOk);
  }
  MINIMAL_CHECK(runner->AllocateTensors() == kTfLiteOk);
}

tflite::SignatureRunner* GetPrefillRunner(
    tflite::Interpreter* interpreter, std::size_t num_input_tokens,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache) {
  // Find the prefill signature that best matches the input token size.
  tflite::SignatureRunner* runner = nullptr;
  int delta = std::numeric_limits<int>::max();
  for (const std::string* key : interpreter->signature_keys()) {
    if (!absl::StrContains(*key, "prefill")) {
      continue;
    }
    TfLiteTensor* input_pos = interpreter->GetSignatureRunner(key->c_str())
                                  ->input_tensor("input_pos");
    // The expected shape for input position is [Seq].
    int seq_size = input_pos->dims->data[0];
    if (num_input_tokens <= seq_size && seq_size - num_input_tokens < delta) {
      runner = interpreter->GetSignatureRunner(key->c_str());
      delta = seq_size - num_input_tokens;
    }
  }
  MINIMAL_CHECK(runner != nullptr);
  PrepareRunner(runner, kv_cache);
  return runner;
}

tflite::SignatureRunner* GetDecodeRunner(
    tflite::Interpreter* interpreter,
    std::map<std::string, std::vector<float, AlignedAllocator<float>>>&
        kv_cache) {
  tflite::SignatureRunner* runner = interpreter->GetSignatureRunner("decode");
  MINIMAL_CHECK(runner != nullptr);
  PrepareRunner(runner, kv_cache);
  return runner;
}

std::unique_ptr<sentencepiece::SentencePieceProcessor>
LoadSentencePieceProcessor() {
  std::ifstream input(absl::GetFlag(FLAGS_sentencepiece_model),
                      std::ios::binary);
  std::string serialized_proto = std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  MINIMAL_CHECK(processor->LoadFromSerializedProto(serialized_proto).ok());
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
  std::map<std::string, std::vector<float, AlignedAllocator<float>>> kv_cache =
      BuildKVCache(interpreter.get());
  MINIMAL_CHECK(!kv_cache.empty())

  // Tokenize the input prompt.
  std::string prompt = absl::GetFlag(FLAGS_prompt);
  std::vector<int> prompt_tokens;
  MINIMAL_CHECK(sp_processor->Encode(prompt, &prompt_tokens).ok());

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

  // Get prefill and decode signature runners.
  std::size_t effective_prefill_token_size = prompt_tokens.size() - 1;
  tflite::SignatureRunner* prefill_runner = GetPrefillRunner(
      interpreter.get(), effective_prefill_token_size, kv_cache);
  MINIMAL_CHECK(prefill_runner != nullptr);
  tflite::SignatureRunner* decode_runner =
      GetDecodeRunner(interpreter.get(), kv_cache);
  MINIMAL_CHECK(decode_runner != nullptr);

  // Get Input Tensors for each of the runners.
  // Shape: [Batch, Seq], Dtype: int32
  TfLiteTensor* prefill_input = prefill_runner->input_tensor("tokens");
  // Shape: [Seq], Dtype: int32
  TfLiteTensor* prefill_input_pos = prefill_runner->input_tensor("input_pos");
  // Shape: [Batch, Seq], Dtype: int32
  TfLiteTensor* decode_input = decode_runner->input_tensor("tokens");
  // Shape: [Seq], Dtype: int32
  TfLiteTensor* decode_input_pos = decode_runner->input_tensor("input_pos");
  // shape: [Batch, kv_cache_max, num_query_groups, head_dim]
  TfLiteTensor* kv_cache_k_0 = decode_runner->input_tensor("kv_cache_k_0");

  int max_seq_size = prefill_input->dims->data[1];
  int kv_cache_max_size = kv_cache_k_0->dims->data[1];

  // Fill in the inputs (assuming one batch).
  // NOTE: We skip the last token and use that during decode.
  int prefill_seq_size =
      std::min(static_cast<int>(prompt_tokens.size()), max_seq_size);
  std::memset(prefill_input->data.i32, 0, prefill_input->bytes);
  std::memset(prefill_input_pos->data.i32, 0, prefill_input_pos->bytes);
  for (int i = 0; i < prefill_seq_size - 1; ++i) {
    prefill_input->data.i32[i] = prompt_tokens[i];
    prefill_input_pos->data.i32[i] = i;
  }
  MINIMAL_CHECK(prefill_runner->Invoke() == kTfLiteOk);

  // Decode until max kv-cache size or user defined step limit, whichever is
  // smaller.
  int max_decode_steps = absl::GetFlag(FLAGS_max_decode_steps) == -1
                             ? kv_cache_max_size
                             : absl::GetFlag(FLAGS_max_decode_steps);
  int decode_steps =
      std::min(max_decode_steps, kv_cache_max_size - prefill_seq_size);
  MINIMAL_CHECK(decode_steps > 0);

  std::vector<int> output_tokens;
  output_tokens.reserve(decode_steps);
  int next_token = prompt_tokens[prefill_seq_size - 1];
  int next_position = prefill_seq_size - 1;
  for (int i = 0; i < decode_steps; ++i) {
    decode_input->data.i32[0] = next_token;
    decode_input_pos->data.i32[0] = next_position;
    MINIMAL_CHECK(decode_runner->Invoke() == kTfLiteOk);
    next_token = GreedySampler(decode_runner->output_tensor("logits"));
    output_tokens.push_back(next_token);
    next_position += 1;
    if (next_token == stop_token_id) {
      break;
    }
  }

  // Detokenize the generated output.
  std::string output_text;
  MINIMAL_CHECK(sp_processor->Decode(output_tokens, &output_text).ok());

  printf("Prompt:\n%s\nOutput text:\n%s\n", prompt.c_str(),
         output_text.c_str());

  return 0;
}
