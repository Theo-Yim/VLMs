# Ovis2.5 LoRA Fine-tuning with TRL

This implementation provides efficient LoRA (Low-Rank Adaptation) fine-tuning for Ovis2.5 multimodal models using Hugging Face TRL library.

## Key Features

- **Memory Efficient**: LoRA reduces memory usage by ~60% compared to full fine-tuning
- **Fast Training**: Only trains adapter layers (1-5% of total parameters)
- **Simple Integration**: Built on TRL SFTTrainer with PEFT support
- **Multimodal Support**: Maintains Ovis2.5's vision-language capabilities
- **Easy Deployment**: Merge adapters or deploy separately

## Files Overview

- `train_theo_lora.py` - Main LoRA training script
- `train_config_lora.json` - LoRA training configuration  
- `merge_lora_adapters.py` - Script to merge adapters with base model
- `train_launch_lora.sh` - Launch script for training

## Requirements

```bash
pip install torch transformers accelerate
pip install trl peft bitsandbytes  # For LoRA support
pip install flash-attn --no-build-isolation  # Optional, for faster attention
```

## Quick Start

### 1. Prepare Your Data

Use the same format as the original Ovis training:

```json
[
  {
    "id": "sample_001",
    "image": "image.jpg",
    "conversations": [
      {
        "from": "human", 
        "value": "<image>\nWhat do you see?"
      },
      {
        "from": "gpt",
        "value": "I see a beautiful landscape..."
      }
    ]
  }
]
```

### 2. Configure Training

Edit `train_config_lora.json`:

```json
{
    "data_path": "./data/train_data.json",
    "image_folder": "./data/images/",
    "output_dir": "./checkpoints/ovis25_lora",
    
    "lora_r": 32,
    "lora_alpha": 64, 
    "lora_dropout": 0.1,
    
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2
}
```

### 3. Run Training

```bash
# Using launch script
./train_launch_lora.sh

# Or directly
python train_theo_lora.py train_config_lora.json
```

### 4. Merge Adapters (Optional)

```bash
python merge_lora_adapters.py \
    --adapter_path ./checkpoints/ovis25_lora \
    --output_path ./checkpoints/ovis25_merged
```

## LoRA Configuration

### Key Parameters

- **`lora_r`** (32): Rank of adaptation matrices. Higher = more parameters but better quality
- **`lora_alpha`** (64): Scaling factor. Usually 2x the rank
- **`lora_dropout`** (0.1): Dropout rate for LoRA layers
- **`lora_target_modules`**: Which layers to adapt

### Target Modules

Default targets Qwen3-8B attention and MLP layers:
```
q_proj, k_proj, v_proj, o_proj,  # Attention
gate_proj, up_proj, down_proj     # MLP
```

For vision components, set `apply_lora_to_vision: true` (experimental).

### Memory vs Quality Trade-offs

| Rank (r) | Parameters | Quality | Memory | Training Time |
|----------|------------|---------|---------|---------------|
| 8        | Very Low   | Good    | Lowest  | Fastest       |
| 16       | Low        | Better  | Low     | Fast          |
| 32       | Medium     | High    | Medium  | Medium        |
| 64       | High       | Highest | Higher  | Slower        |

## Training Tips

### 1. Learning Rate
- LoRA uses higher learning rates (~1e-4) vs full fine-tuning (2e-5)
- Start with 1e-4 and adjust based on loss curves

### 2. Batch Size
- Can use larger effective batch sizes due to lower memory usage
- Try `per_device_train_batch_size=4` with `gradient_accumulation_steps=4`

### 3. Epochs
- LoRA converges faster, typically 1-3 epochs sufficient
- Monitor for overfitting on small datasets

### 4. GPU Memory
- 5k samples should work on 16GB+ GPUs
- Use `gradient_checkpointing=true` to reduce memory further

## Expected Results

For 5k image-text samples on Ovis2.5-9B:

- **Training Time**: ~2-4 hours on RTX 4090
- **GPU Memory**: ~12-16GB
- **Trainable Parameters**: ~67M (1.0% of total)
- **Adapter Size**: ~260MB (vs 18GB full model)

## Deployment Options

### Option 1: Use Adapters Directly
```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("./checkpoints/ovis25_lora")
```

### Option 2: Merge and Deploy
```python
# After merging
from ovis.model.modeling_ovis2_5 import Ovis2_5
model = Ovis2_5.from_pretrained("./checkpoints/ovis25_merged")
```

### Option 3: Multiple Adapters
Keep base model + task-specific adapters for different use cases.

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or use gradient checkpointing
2. **Slow convergence**: Increase learning rate to 2e-4
3. **Import errors**: Install latest TRL and PEFT versions
4. **Quality issues**: Increase LoRA rank or target more modules
5. **Dataloader length error**: Use `max_steps` instead of `num_train_epochs`

### Dataloader Length Issue

If you see: `ValueError: args.max_steps must be set to a positive value if dataloader does not have a length`

**Solution**: Calculate and set `max_steps` explicitly:

```bash
# Calculate steps for your dataset
python calculate_training_steps.py --dataset ./data/train_data.json --batch_size 4 --grad_accum 4 --epochs 5

# Use the output to set max_steps in your config
```

**Formula**: 
- `steps_per_epoch = ceil(dataset_size / (batch_size * grad_accum * num_gpus))`
- `max_steps = steps_per_epoch * num_epochs`

### Memory Optimization

```json
{
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16, 
    "gradient_checkpointing": true,
    "dataloader_num_workers": 2
}
```

## Advanced Usage

### Custom Target Modules
```json
{
    "lora_target_modules": "q_proj,v_proj,o_proj",
    "apply_lora_to_vision": true
}
```

### Quantized Training (QLoRA)
Add to model loading:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

## Comparison with Full Fine-tuning

| Method | Memory | Speed | Quality | Deployment |
|--------|--------|-------|---------|------------|
| Full FT | 40GB+ | Slow | Best | Simple |
| LoRA | 16GB | Fast | Very Good | Flexible |

LoRA is recommended for most use cases due to efficiency and flexibility.