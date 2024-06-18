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

In this section, we will give a more in-depth view for each part of the main components.

### Authoring/Re-authoring library

The user journey of the Edge Generative API begins from the authoring library. The authoring library provides basic building blocks for common GenAI models (from LLMs to diffusion models), and can be mixed together with normal PyTorch ops to compose a generative AI model.

#### Why do we need those building blocks to begin with?

