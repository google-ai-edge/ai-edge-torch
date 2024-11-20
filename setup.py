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

import os
import pathlib
import re

from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = """
Library that supports converting PyTorch models into a .tflite format, which can
then be run with TensorFlow Lite and MediaPipe.  This enables applications for
Android, iOS and IOT that can run models completely on-device.

[Install steps](https://github.com/google-ai-edge/ai-edge-torch#installation)
and additional details are in the AI Edge Torch
[GitHub repository](https://github.com/google-ai-edge/ai-edge-torch).
""".lstrip()

name = "ai-edge-torch"
# TODO(b/357076369): move version updating logics to version.py
version_py = here / "ai_edge_torch" / "version.py"
version_regex = "__version__\s*=\s*(\"|')(?P<version>[^\"']+)(\"|')"
version = re.search(version_regex, version_py.read_text()).group("version")

if nightly_release_date := os.environ.get("NIGHTLY_RELEASE_DATE"):
  name += "-nightly"
  version += ".dev" + nightly_release_date
  version_py.write_text(
      re.sub(
          version_regex, f'__version__ = "{version}"', version_py.read_text()
      )
  )

setup(
    name=name,
    version=version,
    description=(
        "Supporting PyTorch models with the Google AI Edge TFLite runtime."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-ai-edge/ai-edge-torch",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="On-Device ML, AI, Google, TFLite, PyTorch, LLMs, GenAI",
    packages=find_packages(
        include=["ai_edge_torch*"],
    ),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "safetensors",
        "tabulate",
        "torch>=2.4.0",
        "torch_xla>=2.4.0",
        "tf-nightly>=2.19.0.dev20241001",
        "ai-edge-litert-nightly",
        "ai-edge-quantizer-nightly",
    ],
)
