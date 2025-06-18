


# Конвертация модели в tflite

```bash
source venv/bin/activate
cd /home/dragynir/ai_vlm/ai-edge-torch-smalvlm/ai_edge_torch/generative/examples/smalvlm

huggingface-cli download HuggingFaceTB/SmolVLM-256M-Instruct --repo-type model --local-dir ./models/SmolVLM-256M-Instruct


python convert_to_tflite.py --quantize="dynamic_int8"\
 --checkpoint_path='/home/dragynir/ai_vlm/models/SmolVLM-256M-Instruct' --output_path="/home/dragynir/ai_vlm/models/SmolVLM-256M-Instruct-tflite"\
 --mask_as_input=True --prefill_seq_lens=256 --kv_cache_max_len=2048
```

# Конвертация токенайзера
```bash
cd /data/usr/dmitry.korostelev/ml-vlms/mobile/ai-edge-torch/ai_edge_torch/generative/tools
python tokenizer_to_sentencepiece.py \
    --checkpoint=/home/dragynir/ai_vlm/models/SmolVLM-256M-Instruct \
    --output_path=/home/dragynir/ai_vlm/models/SmolVLM-256M-Instruct-tflite/tokenizer.model

Not matched strictly 22/1000 pairs: 2.20%, loosely 6/1000 pairs: 0.60%
```