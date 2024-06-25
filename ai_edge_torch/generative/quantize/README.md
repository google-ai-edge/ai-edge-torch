# Quantization for the AI Edge Torch Generative API

## Typical usage

To apply quantization, we need to create a configuration that fully expresses how the model should be quantized. This configuration is then passed into conversion, generating a quantized model.

`quant_recipes.py` contains a list of recipes that are known to be well-supported during runtime. For the average user, this is a good starting point to select the quantization scheme that is best suited for your deployment needs. After identifying the target recipe, the model can be quantized as follows. This example is extracted from `generative/examples/quantize/example.py`.

```
quant_config = quant_recipes.full_int8_dynamic_recipe()
edge_model = ai_edge_torch.convert(
    model, (tokens, input_pos), quant_config=quant_config
)
```
Once converted, you will get a quantized `.tflite` model which will be ready for on-device deployment.

## Supported schemes

In the current release, the following schemes are supported:

* Dynamic range quantization with FP32 activations and INT8 weights for linear ops
* FP16 quantization with FP16 weights and FP32 activations and computation for all ops

These correspond to the available recipes in `quant_recipes.py`
