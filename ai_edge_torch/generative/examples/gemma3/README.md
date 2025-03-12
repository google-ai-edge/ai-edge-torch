# Gemma 3 Notes

The Gemma 3 is the latest model in the Gemma family of open weights models.

## Tokenizer

The Gemma 3 Tokenizer is available in the Gemma PyTorch repo [here](https://github.com/google/gemma_pytorch). The reauthored models here are compatible with that tokenizer.

## Gemma 3 Task File Creation

Creation of a Task file is needed to use the converted model and tokenizer in the [LLM Inference API](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference). To create the `.task` file for Gemma 3, pip install the [mediapipe](https://pypi.org/project/mediapipe/) Python package and then execute the following Python code:

```python
TFLITE_MODEL = <path to your converted TF Lite model>
TOKENIZER_MODEL = <path to the Gemma 3 tokenizer from the Gemma Pytorch repo>
START_TOKEN="<bos>"
STOP_TOKENS=["<eos>", "<end_of_turn>"]
from mediapipe.tasks.python.genai import bundler
config = bundler.BundleConfig(
    tflite_model=TFLITE_MODEL,
    tokenizer_model=TOKENIZER_MODEL,
    start_token=START_TOKEN,
    stop_tokens=STOP_TOKENS,
    output_filename="/tmp/gemma3.task",
    prompt_prefix="<start_of_turn>user\n",
    prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
)
bundler.create_bundle(config)
```
