#!/bin/bash
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
# A helper script to format code. Must be called from repo's root.
#

# Check if 'gdown' is installed
if ! command -v gdown &> /dev/null
then
    echo "'gdown' is not installed. Installing..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    pip install gdown
else
    echo "Downloading model..."
    file_id="1Yx0gHQs0R9XVM3KleKYsG3iAJrmxIUzP"
    download_url="https://drive.google.com/uc?id=${file_id}"
    destination_file="assets/isnet-general-use.tflite"

    gdown "${download_url}" -O "${destination_file}"
    echo "Download complete!"
fi
