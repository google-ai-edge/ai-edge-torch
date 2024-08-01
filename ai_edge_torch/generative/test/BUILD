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

load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_applicable_licenses = [
        "//third_party/py/ai_edge_torch:license",
    ],
)

py_test(
    name = "test_loader",
    srcs = ["test_loader.py"],
    deps = [
        "//testing/pybase",
        "//third_party/py/ai_edge_torch",
        "//third_party/py/safetensors:torch",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "test_model_conversion",
    srcs = ["test_model_conversion.py"],
    deps = [
        "//testing/pybase",
        "//third_party/py/ai_edge_torch",
        "//third_party/py/numpy",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "test_quantize",
    srcs = ["test_quantize.py"],
    deps = [
        "//testing/pybase",
        "//third_party/py/ai_edge_torch",
        "//third_party/py/parameterized",
        "//third_party/py/torch:pytorch",
    ],
)

py_test(
    name = "test_experimental_ekv",
    srcs = ["test_experimental_ekv.py"],
    deps = [
        "//third_party/py/ai_edge_torch",
        "//third_party/py/torch:pytorch",
    ],
)
