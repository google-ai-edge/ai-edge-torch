# AI Edge Torch Generative API System Architecture Overview

This doc aims to provide a technical deep dive of the AI Edge Torch Generative API, discuss its design considerations, system architecture and major components, current limitations, and future plans for improved system health, usability and performance.

## Main objective

As LLMs/LGMs gain its popularity, we are seeing an increased adoption of this technology in various forms of applications, from serving LLMs in the cloud to deploying smaller language models(SLM) on-device. Running SLMs on-device has benefits such as better privacy protection, faster response time and low maintenance cost. Usually there is a large gap from a trained PyTorch LLM to an efficient and high-performant on-device deployment format, which may include challenging techniques such as model quantization, numerical debug, model quality evaluation, performance optimizations and end-to-end integration with your Android/iOS/Web applications.   

