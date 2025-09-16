# Ovis2.5 Training & Inference Framework

Enhanced training and inference framework for Ovis2.5 multimodal models with DeepSpeed support, evaluation datasets, early stopping, and reflective reasoning capabilities.

## Overview

Comprehensive training pipeline with enhanced features not available in the base framework:

1. **train_theo.py**: Standard Trainer with evaluation dataset support and early stopping
2. **train_theo_trl.py**: TRL SFTTrainer version with advanced training features  
3. **lora/**: LoRA fine-tuning implementation with PEFT integration
4. **utils/**: Cropping tool functionality (work in progress)
5. **train_launch.sh**: DeepSpeed launcher with multi-GPU support

## Usage

### Step 1: Standard Training
Full model fine-tuning with evaluation support:

```bash
# Basic single GPU training
./train_launch.sh ./train_config.json 1

# Multi-GPU training with DeepSpeed
./train_launch.sh ./train_config.json 4
```

### Step 2: TRL Training
SFTTrainer-based training with advanced features:

```bash
./train_launch.sh ./train_config.json 2 trl
```

### Step 3: LoRA Training
Parameter-efficient fine-tuning:

```bash
cd lora/
./train_launch_lora.sh ./train_config_lora.json 1
```

### Step 4: Inference
```python
from inference_ovis25 import Ovis25Inference

ovis = Ovis25Inference("AIDC-AI/Ovis2.5-9B")
response = ovis.single_image_inference(
    "image.jpg", 
    "Describe this image",
    enable_thinking=True
)
```

## Data Format

### Training Dataset
```json
{
  "id": "sample_001",
  "image": "image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat's in this image?"},
    {"from": "gpt", "value": "This image shows..."}
  ]
}
```

### Thinking Mode Training
```json
{
  "from": "gpt", 
  "value": "<think>\nLet me analyze this step by step...\n</think>\n\nThe answer is..."
}
```

## Configuration

### Key Training Parameters
```json
{
  "model_path": "AIDC-AI/Ovis2.5-9B",
  "data_path": "./sample_data/train_data.json", 
  "eval_data_path": "./sample_data/eval_data.json",
  "output_dir": "./checkpoints/ovis25_finetune",
  "deepspeed": "./scripts/zero_configs/zero2_cp.json",
  
  "num_train_epochs": 3,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "learning_rate": 2e-5,
  
  "train_modules": "all",
  "multimodal_max_length": 8192,
  "single_image_max_pixels": 3211264
}
```

### Enhanced Features
- **Automatic step calculation** based on dataset size
- **Early stopping** with evaluation dataset support  
- **DeepSpeed integration** with official zero configs
- **Evaluation dataset loading** with proper error handling
- **Multi-GPU support** with torchrun

## File Structure

```
scripts/
├── run_ovis2_5_sft.sh
└── zero_configs/
    ├── zero0_cp.json         # ZeRO-0 (No Optimization)
    ├── zero1_cp.json         # ZeRO-1 (optimizer state sharding)
    ├── zero2_cp.json         # ZeRO-2 (optimizer + gradient sharding) 
    └── zero3_cp.json         # ZeRO-3 (full parameter sharding)

src_theo/
├── README.md                  # This file
├── train_theo.py              # Standard Trainer with evaluation support
├── train_theo_trl.py          # TRL SFTTrainer version  
├── inference_ovis25.py        # Inference with thinking mode support
├── train_config.json          # Main training configuration
├── train_launch.sh            # DeepSpeed launcher script
├── sample_data/               # Sample data
│   └── data_example.json      # Sample data (not real data)
│   └── train_data.json        # Sample data using sample_small.png
├── lora/                      # LoRA fine-tuning implementation
│   ├── train_theo_lora.py     # LoRA training script
│   ├── train_config_lora.json # LoRA configuration
│   └── train_launch_lora.sh   # LoRA launcher
│   └── merge_lora_adapters.py # LoRA merging utility
└── tools/                     # Tool-related functionality (WIP)
    └── crop_tool.py           # Image cropping utilities
```

## Additional Information about ZeRO Configs

### DeepSpeed ZeRO Optimization Stages

| Stage | Memory Optimization | Communication | Best for |
|-------|-------------------|---------------|----------|
| **ZeRO-0** | No sharding | Minimal | Small models, abundant memory |
| **ZeRO-1** | Optimizer states sharded | Low | Medium models, optimizer bottleneck |
| **ZeRO-2** | Optimizer + gradients sharded | Balanced | **Recommended for Ovis2.5-9B** |
| **ZeRO-3** | Full parameter sharding | High | Very large models, limited memory |

### Memory Reduction vs Baseline

- **ZeRO-0**: 1x (baseline) - Full replication
- **ZeRO-1**: ~4x reduction - Optimizer sharding only
- **ZeRO-2**: ~8x reduction - Optimizer + gradient sharding  
- **ZeRO-3**: Nx reduction - Complete parameter sharding

### Configuration Selection

```bash
# For Ovis2.5-2B (recommended)
"deepspeed": "./scripts/zero_configs/zero1_cp.json"

# For Ovis2.5-9B (current default)
"deepspeed": "./scripts/zero_configs/zero2_cp.json"

# For memory-constrained setups
"deepspeed": "./scripts/zero_configs/zero3_cp.json"

# For small-scale experiments
"deepspeed": "./scripts/zero_configs/zero0_cp.json"
```

### Performance Trade-offs

- **ZeRO-0/1**: Fastest training, highest memory usage
- **ZeRO-2**: Balanced performance and memory efficiency ✅
- **ZeRO-3**: Maximum memory efficiency, slower communication

## Notes

- Models: Ovis2.5-2B/9B with thinking mode support
- Memory: 24GB+ GPU recommended for Ovis2.5-9B
- Features: DeepSpeed, evaluation datasets, early stopping, LoRA support