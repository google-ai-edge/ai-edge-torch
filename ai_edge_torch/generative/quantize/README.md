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

* Dynamic INT8 quantization: FP32 activations, INT8 weights, and integer computation
* Weight-only INT8 quantization: FP32 activations, INT8 weights, and floating point computation
* FP16 quantization: FP16 weights, FP32 activations and floating point computation for all ops
* Dynamic INT4 blockwise quantization: FP32 activations, INT4 weights, and integer computation

 Available preset recipes in [`quant_recipes.py`](./quant_recipes.py).

When using command line, recipes can be specified using strings listed in [`converter.py`](../utilities/converter.py).

For a more concrete example, please see [gemma3](../examples/gemma3/README.md)

### Notes on blockwise quantization
* When blockwise quantization is used, the last dimension of the weight is sliced into blocks of the specific sizes.
* Each block will have its own scale & zero point.
* The smallest block size supported is 32.
* The last dimension shape must also be divisible by the block size specified by users.
* Smaller block sizes are expected to give better model quality, but the model would be bigger and run slower.
* Bigger block sizes are expected to be faster and produce smaller model, but give worse model quality.

## Advanced usage

In addition to configuring quantization using pre-configured recipes in `quant_recipes.py`, users can also customize their recipes according to their specific needs using the `LayerQuantRecipe` and `GenerativeQuantRecipe` API.

`LayerQuantRecipe` specifies at a Generative API layer (`ai_edge_torch/generative/layers`) level how ops within should be quantized. `GenerativeQuantRecipe` specifies at a model level how each component of a Generative API model should be quantized. With these configuration classes, selective quantization can be configured as follows:

```
def custom_selective_quantization_recipe() -> quant_config.QuantConfig:
  return quant_config.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(
          default=create_layer_quant_fp16(),
          embedding=create_layer_quant_int8_dynamic(),
          attention=create_layer_quant_int4_block(32),
          feedforward=create_layer_quant_int4_block(256),
      )
  )
```

For example, this recipe specifies that the embedding table should be quantized to int8, attention should be quantized to int4 with block size of 32, and feedforward layers should be quantized to int4 with block size of 256. All other ops should be quantized to the default scheme which is specified as FP16.


