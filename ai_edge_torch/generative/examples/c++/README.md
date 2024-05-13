# AI Edge Examples

This module offers illustrations of how to utilize and run exported models. The examples provided are designed to be concise and have limited dependencies on third-party libraries. Our intention is for developers to leverage these examples as a starting point for integrating the exported models with their unique model-specific pipelines and requirements.

## Notes:

* If compiling the examples to run on an Android device, you need to download Android NDK and SDK and set `$ANDROID_NDK_HOME` and `$ANDROID_HOME` environment variables. Please note that _bazel_ currently only supports NDK versions 19, 20, and 21.

## Text Generation

In `text_generator_main.cc`, we provide an example of running a decoder-only model end-to-end using TensorFlow Lite as our inference engine.

To get started, you will need an exported  model with two signatures: `prefill` and `decode`. The example takes in an input prompt, tokenizes it, "prefills" the model with the tokens, and decodes autoregressively with greedy sampling until a stop condition is met. Finally, it detokenizes the generated token IDs into text.

It's important to note that while we use [SentencePiece](https://github.com/google/sentencepiece) as the tokenizer module in our example, it's not a requirement, and other tokenizers can be used as needed. Additionally, we're using a greedy sampling strategy, which simply takes an argmax over the output logits. There are many other options available that have been shown to generate better results.

As an example, you can run `text_generator_main`  for an exported Gemma model as follows:

```
bazel run -c opt //ai_edge_torch/generative/examples/c++:text_generator_main -- --tflite_model=PATH/gemma_it.tflite  --sentencepiece_model=PATH/tokenizer.model --start_token="<bos>" --stop_token="<eos>" --num_threads=16 --prompt="Write an email:"
```
