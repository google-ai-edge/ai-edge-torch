# Stable Diffusion through TFLite
This example shows how to use the Edge Generative API to convert a PyTorch Stable Diffusion v1.5 model to TFLite model, and run the image generation inference.

## Convert PyTorch to TFLite model
The example provides two source checkpoints mapping. One is original HuggingFace repo, and the other one is a third party PyTorch implementation of Stable Diffusion. Users need to specify the source checkpoints mapping format for conversion script.

### SafeTensors model checkpoints from original HuggingFace repo
1. Clone original HuggingFace stable diffusion repo [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
2. Run `convert_to_tflite.py` and use `v1-5-pruned-emaonly.safetensors` as the source checkpoints for the conversion script. Notice that optional encoder model is not supported yet.
```bash
python ai_edge_torch/generative/examples/stable_diffusion/convert_to_tflite.py \
--clip_ckpt=$HOME/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
--diffusion_ckpt=$HOME/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
--decoder_ckpt=$HOME/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors \
--output_dir=/tmp/stable_diffusion_safetensors/ \
--ckpt_format=safetensors
```


### PyTorch model checkpoints from third party
1. Download PyTorch stable diffusion model [stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch)
1. Unzip the downloaded model weight into `$HOME/Downloads/stable_diffusion_data`
1. Run `convert_to_tflite.py`. This will convert the PyTorch models into TFLite models. The stable diffusion model has four components: CLIP (text embedding), encoder, diffusion and decoder models. Each component is converted to a single TFLite model file.
```bash
python ai_edge_torch/generative/examples/stable_diffusion/convert_to_tflite.py \
--clip_ckpt=$HOME/Downloads/stable_diffusion_data/ckpt/clip.pt \
--diffusion_ckpt=$HOME/Downloads/stable_diffusion_data/ckpt/diffusion.pt \
--encoder_ckpt=$HOME/Downloads/stable_diffusion_data/ckpt/encoder.pt \
--decoder_ckpt=$HOME/Downloads/stable_diffusion_data/ckpt/decoder.pt \
--output_dir=/tmp/stable_diffusion_pytorch/ \
--ckpt_format=pytorch
```

## Run Stable Diffusion pipeline
1. Use `run_tflite_pipeline` method in `pipeline.py` to trigger the end-to-end stable diffusion pipeline with TFLite model. See the example usage in `pipeline.py` as a script.

```bash
python ai_edge_torch/generative/examples/stable_diffusion/pipeline.py \
--tokenizer_vocab_dir=$HOME/stable-diffusion-v1-5/tokenizer/ \
--clip_ckpt=/tmp/stable_diffusion_safetensors/clip.tflite \
--diffusion_ckpt=/tmp/stable_diffusion_safetensors/diffusion.tflite \
--decoder_ckpt=/tmp/stable_diffusion_safetensors/decoder.tflite \
--output_path=/tmp/sd_result_tflite.jpg \
--n_inference_steps=20
```

Here is an example generated image.

Prompt: "a photograph of an astronaut riding a horse"

![](sd_result_tflite.jpg)