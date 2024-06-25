# AI Edge Torch Generative API System Architecture Overview

This doc aims to provide a technical deep dive of the AI Edge Torch Generative API, discuss its design considerations, system architecture and major components, current limitations, and future plans for improved system health, usability and performance.

## Main objective

As LLMs/LGMs gain its popularity, we are seeing an increased adoption of this technology in various forms of applications, from serving LLMs in the cloud to deploying smaller language models(SLM) on-device. Running SLMs on-device has benefits such as better privacy protection, faster response time and low maintenance cost. Usually there is a large gap from a trained PyTorch LLM to an efficient and high-performant on-device deployment format, which may include challenging techniques such as model quantization, numerical debug, model quality evaluation, performance optimizations and end-to-end integration with the production environment (e.g. Android/iOS apps, or web). To overcome those issues, we introduced the AI Edge Torch Generative API, which aims to provide a complete end-to-end solution to solve the deployment painpoints. The library is still under active development and the API interface may change in the coming months.

## Overall System Architecture

On a high level, the library is composed of the following major components:
1) Authoring/Re-authoring library (with initial support in PyTorch).
2) Checkpoint loading APIs (weight remapping).
3) Quantization APIs.
4) Multi-signature conversion API.
5) Composite op lowering via high-level function boundary.
6) End-to-end Pipeline APIs (MediaPipe-based or pure C++ inference).
7) On-device execution via TF Lite and delegate system.
8) Model visualization and profiling tools.

TODO(haoliang): Draw a nice picture here with all the necessary components.

## Generative API User Journey

TODO(haoliang): Draw another nice diagram showing the user journey of Generative API.

## Detailed design

In this section, we will give a more in-depth view for the main components of the system.

### Authoring/Re-authoring library

The user journey of the Edge Generative API begins from the authoring library. The authoring library provides basic building blocks for common GenAI models (from LLMs to diffusion models), and users can leverage those building blocks to author their own custom generative models and achieve great out-of-the-box performance. 

#### Motivation for the PyTorch building blocks

During our initial investigation, we found that although there are many open-source PyTorch implementation of common LLM models (e.g. Llama), it's pretty difficult to convert those models from source to TF Lite format, and achieve good performance. Those difficulties might include:

1) Not able to be exported by Torch Dynamo. The PyTorch 2.0 compiler uses torch dynamo under-the-hood to perform graph capturing (tracing a PyTorch `nn.Module` and convert it to a full graph-like representation in FX Graph). For an arbitrary OSS model, dynamo may fail to export the full graph and fail in the middle. Re-writing the model to be dynamo exportable sometimes may require a non-trivial rewriting of the model logic.
2) Not able to be converted by AI edge torch. AI edge torch converter utilizes TorchXLA, StableHLO and TF Lite converter to convert the FX graph to TF Lite model format. During this process, it may fail if the source FX graph has certain graph patterns / ops that are not supported by our converter. As the conversion will go through multiple stages, it's very difficult to triage and fix the conversion issue (either fixing the model itself or converter stack) for most users.
3) Difficult to guarantee performance. Even you get lucky and successfully make it through the TF Lite model format, there is also no guarantee that the model will run performantly on device, and be able to leverage on device accelerators such as XNNPack, GPU or NPUs. For example, performance critical layers such as KV Cache and SDPA need to be handlded specifically in the converter and runtime to ensure it can be accelerated. Without custom designed building blocks, the converted models won't run very efficiently.
4) Applying quantization consistently is difficult. It's difficult to specify quantization receipes in a consistent way for an arbitrary PyTorch LLM model. For ODML, we need to co-design our custom building blocks and AI Edge quantizer to ensure they work smoothly together.
5) Many OSS models are not designed / optimized for mobile. For example, those implementations may contain distributed training logic, CUDA dependencies, and tensor parallel optimizations for training, which make exporting to mobile runtime extremely difficult without a significant re-write of the model.

We also noticed solutions such as `llama.cpp`, or `gemma.cpp` which allow you to directly write the model inference logic in C++. Those solutions also become very popular in the OSS community since users can just code the model in C++ without much dependencies, and it achieves great performance. However, those solutions may suffer from:
1) Poor debuggability. It's not easy to step into the C++ inference logic to debug numerical issues, or performance issues. It usually requires a lot of manual debugging work without enough tooling support. Also model visualization tools (e.g. Model Explorer, or Netron) won't work for those solutions.
2) Requires deep expertise in runtime and low-level kernel optimizations. E.g. models written this way usually has close coupling with kernel libraries (e.g. Highway, XNNPack, OpenCL etc) which are diffcult for most mobile ML practitioners. You will need to have sufficient knowledge in those kernel libraries to make any performance improvements. We find that there is a steep learning curve for most developers to get started on contributing.
3) Difficult to scale to new models / architectures. Since those solutions apply the "model as code" idea, the implementation itself is usually very customized to a specific model architecture (e.g. Llama2) and hard to generalize to new models in the future. The flexibility to configure and customize is usually very limited.
4) Difficult to support accelerators. LLMs and Generative models usually require GPU/NPU/TPU acceleration to ensure they can achieve the best performance and power efficiency. It's especially critical to leverage those acclerators in mobile environments since phones/tablets/edge devices are very resource contrained. We need a generalized approach to separate model representation and on-device acceleration to minimize the amount of work for any new NPU support.

With those considerations and observations, we propose a set of PyTorch layers which implement core functionality of Generative AI models, with highly customized optimizations for mobile deployments. Those layers provide nice properties such as:
1) Easy to compose. Since they are implemented using native PyTorch ops and APIs, most PyTorch users can understand and compose with them with less effort.
2) Easy to debug. We provide tooling such as Model explorer to allow visualization of the ExportedProgram or TF Lite models. It's much easier to understand model architecture, and layer composition with the help of those tools.
3) Convertibility guarantee. Since we carefully designed and implemented those building blocks, we can ensure that models composed via those building blocks can convert to TF Lite without additional manual work.
4) Performance guarantee. Those layers are implemented to ensure best performance on mobile hardwares (CPU, GPU, NPUs), and they carry high-level function boundary information by default to leverage highly performant fused kernels.
5) Easy to quantize. Paired with the AI Edge quantization APIs, users can easily specify custom quantization receipes for the models.
6) Scalability to new models and architectures. The layers are designed with cleanly separated interfaces, and are highly reusable and customizable to support any new Generative models. Users can also contribute new building blocks to the layer library to support any missing features.
7) Easy to support ML accelerators. Since layers are implemented with portability in-mind, it's also easy to perform backend-specific graph mutations (e.g. via FX passes) to ensure the models are friendly for ML accelerators. HLFB can also play a critical role here to ensure PyTorch ops are mapped to the most efficient kernels supplied by different backends.

For more documentation on the layer design, please refer to [this page](https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative/layers).

### Checkpoint loading APIs.



### Quantization via AI Edge quantizer
TODO(psho): fill in this part.

