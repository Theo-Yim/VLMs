# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Vision-Language Models (VLMs) research repository containing implementations, training scripts, and inference code for multiple multimodal large language models:

- **Ovis** - Open VISion model with structural embedding alignment
- **InternVL3** - International Vision-Language model series
- **Custom Trainers** - (Not used) Specialized training implementations for Ovis2.5 and QwenVL2.5
- **Utils** - Supporting utilities and tools

## Architecture

### Main Components

- `Ovis/` - Complete Ovis model implementation with HuggingFace integration, training scripts, and inference
- `InternVL3/` - InternVL3.5 models with LoRA fine-tuning support and RefCOCO evaluation
- `custom_trainer/` - Enhanced training implementations with two-stage training (SFT + R-GRPO)
- `utils/` - Shared utilities including LH_POC-related files and video frame extraction and experimental code

### Model-Specific Architectures

**Ovis2.5:**
- Uses `AutoModelForCausalLM` (not `AutoModelForVision2Seq`)
- Native resolution processing via NaViT
- Requires batch_size=1 due to variable tensor sizes
- Custom preprocessing with `model.preprocess_inputs()`
- Advanced thinking mode with budget control

**InternVL3.5:**
- Standard HuggingFace Vision2Seq architecture
- Dynamic resolution support
- LoRA fine-tuning compatible
- RefCOCO grounding evaluation support

## Key Training Commands

### Ovis Training

```bash
# Official training script
bash Ovis/scripts/run_ovis2_5_sft.sh

# Custom trainer
bash Ovis/src_theo/tran_launch.sh

# LoRA training (recommended for memory efficiency)
bash Ovis/src_theo/lora/train_launch_lora.sh
```

### InternVL3 Training

```bash
# LoRA fine-tuning
cd InternVL3/src_theo/lora
python train_theo_lora.py train_config.json

# Multi-GPU training
bash train_launch_lora.sh
```

## Inference Commands

### Ovis Inference

```bash
# Basic inference
python ovis/serve/infer_basic_demo.py

# Thinking mode (reflective reasoning)
python ovis/serve/infer_think_demo.py

# Web UI
python ovis/serve/web_ui.py --model-path AIDC-AI/Ovis2.5-9B --port 8001

# vLLM server
vllm serve AIDC-AI/Ovis2.5-9B --trust-remote-code --port 8000
```

### InternVL3 Inference

```bash
# Batch inference
python InternVL3/inference_img_batch.py

# Video inference
python InternVL3/inference_vd.py

# Simple inference
python InternVL3/inference_simple.py
```

## Data Formats

### Ovis Training Data (JSONL)
```json
{
  "conversations": [
    {
      "id": "sample_001",
      "image": "cat_on_table.jpg",
      "conversations": [
        {
          "from": "human",
          "value": "<image>\nWhat do you see in this image?"
        },
        {
          "from": "gpt",
          "value": "I see a beautiful orange tabby cat sitting on a wooden table. The cat appears to be relaxed and is looking directly at the camera. The background shows a cozy indoor setting with soft lighting."
        }
      ]
    },
}
```

### InternVL3 Training Data (JSON)
```json
[
  {
    "id": "sample_001",
    "image": "image.png",
    "conversations": [
      {"from": "human", "value": "<image>\nQuestion"},
      {"from": "gpt", "value": "Response"}
    ]
  }
]
```

## Testing and Validation

```bash
# Test Ovis integration
cd custom_trainer/Ovis2.5
python test_ovis_integration.py

# Test InternVL inference
cd InternVL3/src_theo/lora
python test_inference.py

# RefCOCO evaluation
cd InternVL3/refcoco
python evaluate_refcoco.py
```

## Development Notes

- All models support thinking mode with `<think>` tags for reasoning
- Grounding support via `<ref>`, `<box>`, `<point>` tags (Ovis) or tool calls (custom trainer)
- Flash attention recommended for performance: `pip install flash-attn --no-build-isolation`
- Use BF16 (`--bf16`) for better stability than FP16
- Monitor training with TensorBoard logs in respective output directories