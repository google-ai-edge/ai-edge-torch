#!/usr/bin/env bash
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/..
cd "${ROOT_DIR}"

build_configs=("linux" "android_arm64")

echo "NDK PATH = ${ANDROID_NDK_HOME}"
echo "SDK PATH = ${ANDROID_HOME}"
echo "Current working directory: $(pwd)"

FAILED=false
for cfg in "${build_configs[@]}"
do
  echo "Build config = ${cfg}"
  BUILD_COMMAND="bazel build -c opt --config=${cfg} //ai_edge_torch/generative/examples/cpp:text_generator_main"
  echo "Build command: ${BUILD_COMMAND} "
  ${BUILD_COMMAND}
  if [ $? == 0 ]; then
    echo "Build succeeded :)"
  else
    echo "Build failed..."
    FAILED=true
  fi
done

if [[ ${FAILED} = true ]]
then
  exit 1
fi
