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
load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = [
        "//third_party/py/ai_edge_torch:license",
    ],
    default_visibility = [
        "//third_party/py/ai_edge_torch:__subpackages__",
    ],
)

py_library(
    name = "llama",
    srcs = ["llama.py"],
    deps = [
        "//third_party/py/ai_edge_torch",
        "//third_party/py/ai_edge_torch/generative/examples/tiny_llama",
        "//third_party/py/ai_edge_torch/generative/utilities:loader",
        "//third_party/py/torch:pytorch",
    ],
)

py_binary(
    name = "convert_to_tflite",
    srcs = ["convert_to_tflite.py"],
    python_version = "PY3",
    deps = [
        ":llama",
        "//third_party/py/ai_edge_torch",
    ],
)

py_binary(
    name = "verify",
    srcs = ["verify.py"],
    python_version = "PY3",
    deps = [
        ":llama",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/ai_edge_torch/generative/utilities:transformers_verifier",
        "//third_party/py/ai_edge_torch/generative/utilities:verifier",
        "//third_party/py/transformers",
    ],
)
