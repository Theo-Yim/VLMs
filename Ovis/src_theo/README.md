# Ovis2.5 Training & Inference Framework

Enhanced training pipeline for Ovis2.5 multimodal models with tool-calling support, DeepSpeed optimization, and GRPO refinement.

## Directory Structure

```
src_theo/
├── sft/                      # Supervised Fine-Tuning
│   ├── train_theo.py        # Standard Trainer (recommended)
│   ├── train_theo_trl.py    # TRL SFTTrainer variant
│   ├── train_config.json    # Training configuration
│   └── train_launch.sh      # Multi-GPU launcher with DeepSpeed
│
├── lora/                     # LoRA Fine-Tuning
│   ├── train_theo_lora.py   # LoRA training script
│   ├── train_config_lora.json
│   ├── train_launch_lora.sh
│   └── merge_lora_adapters.py
│
├── grpo/                     # GRPO Refinement (after SFT)
│   ├── train_theo_grpo.py   # GRPO training script
│   ├── ovis_grpo_trainer.py # Custom multimodal GRPO trainer
│   ├── ovis_grpo_generator.py
│   ├── grpo_config_tool.json
│   ├── train_launch_grpo.sh
│   └── GRPO_TRAINING_README.md
│
├── tools/                    # Tool System (Crop, Identify)
│   ├── tool_base.py         # ToolBase + ToolRegistry
│   ├── crop_tool.py         # Crop tool implementation
│   ├── mock_id_tool.py      # Identify tool (mock)
│   ├── inference_integration.py
│   └── README_TOOL_SYSTEM.md
│
├── inference_ovis25.py       # Inference wrapper
└── sample_data/              # Example data
```

## Quick Start

### 1. Standard SFT Training
```bash
# Single GPU
cd /workspace/VLMs
bash Ovis/src_theo/sft/train_launch.sh Ovis/src_theo/sft/train_config.json 1

# Multi-GPU with DeepSpeed ZeRO-2
bash Ovis/src_theo/sft/train_launch.sh Ovis/src_theo/sft/train_config.json 4
```

### 2. LoRA Training (Memory Efficient)
```bash
cd /workspace/VLMs
bash Ovis/src_theo/lora/train_launch_lora.sh
```

### 3. GRPO Refinement (Optional)
```bash
# After SFT completes
cd /workspace/VLMs
bash Ovis/src_theo/grpo/train_launch_grpo.sh
```

## Training Methods

| Method | Use When | Memory | Speed |
|--------|----------|--------|-------|
| **SFT** | Initial training on supervised data | High | Fast |
| **LoRA** | Limited GPU memory | Low | Fastest |
| **GRPO** | Refining tool usage after SFT | High | Slow |

## Key Features

- **Tool-calling support**: Crop and Identify tools with automatic execution
- **Tool response masking**: Prevents hallucination during training
- **DeepSpeed integration**: ZeRO-0/1/2/3 optimization stages
- **Early stopping**: Automatic training termination on eval dataset
- **Thinking mode**: Reflective reasoning with `<think>` tags
- **Multi-GPU**: DDP and DeepSpeed distributed training

## Data Format

### Standard Conversation
```json
{
  "image": "train2017/000000123.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat's in this image?"},
    {"from": "gpt", "value": "The image shows..."}
  ]
}
```

### Tool-Calling Format
```json
{
  "image": "train2017/000000456.jpg",
  "conversations": [
    {"from": "system", "value": "You have access to Crop and Identify tools..."},
    {"from": "human", "value": "<image>\nWho is the person on the left?"},
    {"from": "gpt", "value": "<think>\nLet me crop the person first.\n<tool_call>Crop [10,20,100,200]</tool_call><image>\nNow identify them.\n<tool_call>Identify [10,20,100,200]</tool_call><tool_response>John Doe</tool_response>\n</think>\n<answer>The person on the left is John Doe.</answer>"}
  ]
}
```

## Configuration

### SFT Config (`sft/train_config.json`)
```json
{
  "model_path": "AIDC-AI/Ovis2.5-9B",
  "data_path": "./data/train_data.json",
  "output_dir": "./checkpoints/ovis25_finetune",
  "deepspeed": "./Ovis/scripts/zero_configs/zero2_cp.json",

  "num_train_epochs": 2,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 32,
  "learning_rate": 5e-5
}
```

### LoRA Config (`lora/train_config_lora.json`)
```json
{
  "lora_r": 32,
  "lora_alpha": 64,
  "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
  "learning_rate": 1e-4
}
```

### GRPO Config (`grpo/grpo_config_tool.json`)
```json
{
  "sft_model_path": "./Ovis/checkpoints/ovis25_finetune_final",
  "num_generations": 4,
  "tool_usage_weight": 0.4,
  "bbox_validity_weight": 0.3,
  "reasoning_quality_weight": 0.3
}
```

## DeepSpeed ZeRO Stages

| Stage | Memory Reduction | Best For |
|-------|-----------------|----------|
| ZeRO-0 | None | Small models, debugging |
| ZeRO-1 | ~4x | Medium models |
| ZeRO-2 | ~8x | **Ovis2.5-9B (recommended)** |
| ZeRO-3 | Nx | Very large models, limited VRAM |

## Tool System

See `tools/README_TOOL_SYSTEM.md` for details on:
- Creating custom tools
- Tool execution during inference
- Tool response masking during training
- Image-returning vs text-returning tools

## Inference

```python
from inference_ovis25 import Ovis25Inference

ovis = Ovis25Inference("AIDC-AI/Ovis2.5-9B")
response = ovis.single_image_inference(
    "image.jpg",
    "Describe this image",
    enable_thinking=True
)
```

## Notes

- Ovis2.5 requires `batch_size=1` due to variable tensor sizes (NaViT)
- Use BF16 (`bf16: true`) for better stability
- Flash attention recommended: `pip install flash-attn --no-build-isolation`
- Tool responses are masked from loss computation during training
