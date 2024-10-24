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
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    default_applicable_licenses = [
        "//third_party/py/ai_edge_torch:license",
    ],
)

py_library(
    name = "sampler",
    srcs = ["sampler.py"],
    deps = ["//third_party/py/numpy"],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":k_euler",
        ":k_euler_ancestral",
        ":k_lms",
    ],
)

py_library(
    name = "k_euler_ancestral",
    srcs = ["k_euler_ancestral.py"],
    deps = [
        ":sampler",
        "//third_party/py/ai_edge_torch/generative/examples/stable_diffusion:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "k_lms",
    srcs = ["k_lms.py"],
    deps = [
        ":sampler",
        "//third_party/py/ai_edge_torch/generative/examples/stable_diffusion:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "k_euler",
    srcs = ["k_euler.py"],
    deps = [
        ":sampler",
        "//third_party/py/ai_edge_torch/generative/examples/stable_diffusion:util",
        "//third_party/py/numpy",
    ],
)
