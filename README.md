# AI Edge Torch

AI Edge Torch is a python library that supports converting PyTorch models into a
.tflite format, which can then be run with TensorFlow Lite and MediaPipe.
This enables applications for Android, iOS and IOT that can run models
completely on-device. AI Edge Torch offers broad CPU coverage, with initial GPU
and NPU support.  AI Edge Torch seeks to closely integrate with PyTorch,
building on top of torch.export() and providing good coverage of Core ATen
operators.

To get started converting PyTorch models to TF Lite, see additional details in
the [PyTorch converter](#pytorch-converter) section. For the particular case of
Large Language Models (LLMs) and transformer-based models, the [Generative
API](#generative-api) supports model authoring and quantization to enable
improved on device performance.

Although part of the same PyPi package, the PyTorch converter is a Beta release,
while the Generative API is an Alpha release. Please see the [release
notes](https://github.com/google-ai-edge/ai-edge-torch/releases/) for additional
information.

## PyTorch Converter
Here are the steps needed to convert a PyTorch model to a TFLite flatbuffer:

```python
import torch
import torchvision
import ai_edge_torch

# Use resnet18 with pre-trained weights.
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
sample_inputs = (torch.randn(1, 3, 224, 224),)

# Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
# are setting the model to evaluation mode prior to conversion.
edge_model = ai_edge_torch.convert(resnet18.eval(), sample_inputs)
edge_model.export("resnet18.tflite")
```

The [getting started](docs/pytorch_converter/getting_started.ipynb) Jupyter
notebook gives an initial walkthrough of the conversion process and can be tried
out with Google Colab.

Additional technical details of the PyTorch Converter are [here](docs/pytorch_converter/README.md).

## Generative API
The AI Edge Torch Generative API is a Torch native library for authoring
mobile-optimized PyTorch Transformer models, which can be converted to TFLite,
allowing users to easily deploy Large Language Models (LLMs) on mobile
devices. Users can convert the models using the AI Edge Torch PyTorch
Converter, and run them via the TensorFlow Lite runtime. See
[here](ai_edge_torch/generative/examples/cpp).

Mobile app developers can also use the Edge Generative API to integrate PyTorch
LLMs directly with the MediaPipe LLM Inference API for easy integration within
their application code. See
[here](http://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference#ai_edge_model_conversion).

More detailed documentation can be found [here](ai_edge_torch/generative).

The Generative API is currently CPU-only, with planned support for GPU and NPU.
A further future direction is to collaborate with the PyTorch community to
ensure that frequently used transformer abstractions can be directly supported
without reauthoring.


## Build Status

Build Type         |    Status     |
-----------        | --------------|
Generative API (Linux) | [![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_generative_api.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_generative_api.yml) |
Model Coverage (Linux) | [![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_model_coverage.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_model_coverage.yml) |
Unit tests (Linux)     | [![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_unittests.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_unittests.yml) |
Nightly Release    | [![](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_release.yml/badge.svg?branch=main)](https://github.com/google-ai-edge/ai-edge-torch/actions/workflows/nightly_release.yml) |

## Installation

### Requirements and Dependencies

 * Python versions: >=3.10
 * Operating system: Linux
 * PyTorch: [![torch](https://img.shields.io/badge/torch->=2.4.0-blue)](https://pypi.org/project/torch/)
 * TensorFlow: [![tf-nightly](https://img.shields.io/badge/tf--nightly-latest-blue)](https://pypi.org/project/tf-nightly/)

<!-- requirement badges are updated by ci/update_nightly_versions.py -->

### Python Virtual Env

Set up a Python virtualenv:
```bash
python -m venv --prompt ai-edge-torch venv
source venv/bin/activate
```

The latest stable release can be installed with:
```bash
pip install ai-edge-torch
```

Alternately, the nightly version can be installed with:
```bash
pip install ai-edge-torch-nightly
```

### Update LD_LIBRARY_PATH if necessary

Torch XLA builds a shared library, `_XLAC.so` that needs to link to the version of Python
it was built with (currently 3.10 or 3.11). In order to ensure that `import _XLAC` can succeed,
update the LD_LIBRARY_PATH to the lib directory of your Python environment:

```bash
export LD_LIBRARY_PATH=<path to Python installation>/lib:$LD_LIBRARY_PATH
```


* The list of versioned releases can be seen [here](https://github.com/google-ai-edge/ai-edge-torch/releases).
* The full list of PyPi releases (including nightly builds) can be seen [here](https://pypi.org/project/ai-edge-torch/#history).


# Contributing

See our [contribution documentation](CONTRIBUTING.md).

# Getting Help

Please [create a GitHub issue](https://github.com/google-ai-edge/ai-edge-torch/issues/new/choose) with any questions.
