
# Convert model to tflite

```bash
cd ai-edge-torch
python -m venv venv
source venv/bin/activate
pip install -r dev-requirements.txt
pip install -e .


cd ai_edge_torch/generative/examples/smalvlm
huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct --repo-type model --local-dir ./models/SmolVLM-256M-Instruct


python convert_to_tflite.py --quantize="dynamic_int8"\
 --checkpoint_path='./models/SmolVLM-256M-Instruct' --output_path="./models/SmolVLM-256M-Instruct-tflite"\
 --mask_as_input=True --prefill_seq_lens=256 --kv_cache_max_len=2048
```

# Test tflite model with Tflite Python Interpreter

```bash
python test_tflite.py
```

# Convert model tokenizer
```bash
cd ai_edge_torch/generative/tools
python tokenizer_to_sentencepiece.py \
    --checkpoint=YOUR_PATH/models/SmolVLM-256M-Instruct \
    --output_path=YOUR_PATH/models/SmolVLM-256M-Instruct-tflite/tokenizer.model

Not matched strictly 22/1000 pairs: 2.20%, loosely 6/1000 pairs: 0.60%
```

# Verify model
You can also run verify scripts, before converting model to tflite, to check reauthored model

```bash
python verify.py
python verify_text.py
python verify_image.py
```
