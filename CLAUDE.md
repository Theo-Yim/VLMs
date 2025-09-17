# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Vision-Language Models (VLMs) research repository containing implementations, training scripts, and inference code for multiple multimodal large language models:

- **Ovis** - Open VISion model with structural embedding alignment
- **InternVL3** - International Vision-Language model series
- **Custom Trainers** - Specialized training implementations for Ovis2.5 and QwenVL2.5
- **Utils** - Supporting utilities and tools

## Architecture

### Main Components

- `Ovis/` - Complete Ovis model implementation with HuggingFace integration, training scripts, and inference
- `InternVL3/` - InternVL3.5 models with LoRA fine-tuning support and RefCOCO evaluation
- `custom_trainer/` - Enhanced training implementations with two-stage training (SFT + R-GRPO)
- `utils/` - Shared utilities including video frame extraction and experimental code

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
# Install Ovis dependencies
cd Ovis
pip install -r requirements.txt
pip install -e .

# Official training script
bash scripts/run_ovis2_5_sft.sh

# Custom trainer with both SFT and R-GRPO stages
cd custom_trainer/Ovis2.5
python train_ovis25.py --stage both --use_lora --bf16

# LoRA training (recommended for memory efficiency)
python train_ovis25.py --use_lora --lora_r 128 --lora_alpha 256
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

## Critical Requirements

### Ovis2.5 Constraints
- **MUST use batch_size=1** - Native resolution processing prevents batching
- Use `gradient_accumulation_steps` for effective batch size
- LoRA training strongly recommended for memory efficiency
- Requires `lora_patch.py` for LoRA compatibility

### Memory Requirements
- Ovis2.5 inference: ~18GB VRAM
- Ovis2.5 LoRA training: ~24-28GB VRAM
- InternVL3 LoRA training: ~16-24GB VRAM depending on batch size

## Data Formats

### Ovis Training Data (JSONL)
```json
{
  "image_path": "path/to/image.jpg",
  "QnA": [
    {
      "Q": "Question text",
      "A3": "<think>reasoning</think>\n<answer>response</answer>"
    }
  ]
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