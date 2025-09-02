# Ovis2.5 LoRA Fine-tuning with TRL

This implementation provides efficient LoRA (Low-Rank Adaptation) fine-tuning for Ovis2.5 multimodal models using Hugging Face TRL library.

## Key Features

- **Memory Efficient**: LoRA reduces memory usage by ~60% compared to full fine-tuning
- **Fast Training**: Only trains adapter layers (1-5% of total parameters)
- **Simple Integration**: Built on TRL SFTTrainer with PEFT support
- **Multimodal Support**: Maintains Ovis2.5's vision-language capabilities
- **Multi-GPU Support**: Automatically handles single and multi-GPU training
- **Easy Deployment**: Merge adapters or deploy separately

## Files Overview

- `train_theo_lora.py` - Main LoRA training script
- `train_config_lora.json` - Standard LoRA training configuration  
- `train_config_lora_optimized.json` - Optimized configuration with validation
- `train_launch_lora.sh` - Launch script for easy training
- `merge_lora_adapters.py` - Script to merge adapters with base model
- `split_dataset.py` - Utility to split dataset into train/validation sets

## Requirements

```bash
pip install torch transformers accelerate
pip install trl peft bitsandbytes  # For LoRA support
pip install flash-attn --no-build-isolation  # Optional, for faster attention
```

## Quick Start

### 1. Prepare Your Data

Use the Ovis conversation format:

```json
[
  {
    "id": "sample_001",
    "image": "image.jpg",
    "conversations": [
      {
        "from": "human", 
        "value": "<image>\nWhat do you see in this construction site?"
      },
      {
        "from": "gpt",
        "value": "I can see a construction site with workers wearing safety helmets..."
      }
    ]
  }
]
```

### 2. Split Your Dataset (Optional)

If you have a single dataset file, split it into training and validation sets:

```bash
python split_dataset.py
```

This will create `train_[original_name].json` and `validation_[original_name].json` files.

### 3. Configure Training

Choose one of the provided configurations or create your own:

#### Standard Configuration (`train_config_lora.json`)
- Basic LoRA setup without validation
- Good for initial experiments

#### Optimized Configuration (`train_config_lora_optimized.json`)
- Includes validation dataset
- Better hyperparameters for production training
- Early stopping and best model selection

Edit your chosen config file to update paths:

```json
{
    "data_path": "./utils/lh-poc/train_training_dataset_lh_jh.json",
    "eval_data_path": "./utils/lh-poc/val_training_dataset_lh_jh.json",
    "image_folder": "/path/to/your/images",
    "output_dir": "./Ovis/checkpoints/your_model_name",
    
    "lora_r": 64,
    "lora_alpha": 128, 
    "lora_dropout": 0.05,
    
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4
}
```

### 4. Run Training

#### Easy Launch with Script

```bash
# Use default configuration (train_config_lora.json) and 1 GPU
sh train_launch_lora.sh

# Specify configuration file and use 8 GPUs
sh train_launch_lora.sh "./Ovis/src_theo/lora/train_config_lora_optimized.json" 8

# Specify just configuration file (uses 1 GPU by default)
sh train_launch_lora.sh "./Ovis/src_theo/lora/train_config_lora_optimized.json"
```

#### Direct Python Call

```bash
# Single GPU
python ./Ovis/src_theo/lora/train_theo_lora.py "./Ovis/src_theo/lora/train_config_lora_optimized.json"

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 ./Ovis/src_theo/lora/train_theo_lora.py "./Ovis/src_theo/lora/train_config_lora_optimized.json"
```

### 5. Merge Adapters (Optional)

After training, you can merge the LoRA adapters with the base model:

```bash
python merge_lora_adapters.py \
    --adapter_path ./Ovis/checkpoints/your_model_name \
    --output_path ./Ovis/checkpoints/your_model_merged
```

## LoRA Configuration

### Key Parameters

- **`lora_r`** (32-128): Rank of adaptation matrices. Higher = more parameters but better quality
- **`lora_alpha`** (64-256): Scaling factor. Usually 2x the rank
- **`lora_dropout`** (0.05-0.1): Dropout rate for LoRA layers
- **`lora_target_modules`**: Which layers to adapt

### Target Modules

Default targets attention and MLP layers:
```
q_proj, k_proj, v_proj, o_proj,  # Attention
gate_proj, up_proj, down_proj     # MLP
```

For vision components, set `apply_lora_to_vision: true` (experimental).

### Recommended Configurations

#### Small Dataset (<1k samples)
```json
{
    "lora_r": 32,
    "lora_alpha": 64,
    "learning_rate": 1e-4,
    "num_train_epochs": 3
}
```

#### Medium Dataset (1k-10k samples)
```json
{
    "lora_r": 64,
    "lora_alpha": 128,
    "learning_rate": 2e-4,
    "num_train_epochs": 2
}
```

#### Large Dataset (>10k samples)
```json
{
    "lora_r": 128,
    "lora_alpha": 256,
    "learning_rate": 1e-4,
    "num_train_epochs": 1
}
```

## Training Tips

### 1. Learning Rate
- LoRA typically uses higher learning rates (1e-4 to 2e-4)
- Start with 2e-4 for most cases
- Use cosine scheduler for better convergence

### 2. Batch Size and Memory
- Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`
- For 24GB GPU: batch_size=4, grad_accum=1
- For 16GB GPU: batch_size=2, grad_accum=2
- Enable `gradient_checkpointing` to save memory

### 3. Epochs and Steps
- LoRA converges faster than full fine-tuning
- Use validation dataset to monitor convergence
- Enable early stopping to prevent overfitting

### 4. Multi-GPU Training
- The launch script automatically detects available GPUs
- Uses `torchrun` for multi-GPU training
- Reduces training time significantly

## Expected Results

For 5k image-text samples on Ovis2.5-9B:

### Single GPU (RTX 4090)
- **Training Time**: ~4-6 hours
- **GPU Memory**: ~16-20GB
- **Trainable Parameters**: ~67M (1.0% of total)

### 8x GPU Setup
- **Training Time**: ~30-60 minutes
- **GPU Memory**: ~12-16GB per GPU
- **Adapter Size**: ~260MB (vs 18GB full model)

## Deployment Options

### Option 1: Use Adapters Directly
```python
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "./Ovis/checkpoints/your_model_name",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Option 2: Merge and Deploy
```python
# After merging with merge_lora_adapters.py
from ovis.model.modeling_ovis2_5 import Ovis2_5
model = Ovis2_5.from_pretrained("./Ovis/checkpoints/your_model_merged")
```

### Option 3: Multiple Task-Specific Adapters
Keep base model + different adapters for different tasks/domains.

## Troubleshooting

### Common Issues

1. **CUDA OOM**: 
   - Reduce `per_device_train_batch_size`
   - Enable `gradient_checkpointing`
   - Increase `gradient_accumulation_steps`

2. **Slow convergence**: 
   - Increase learning rate to 2e-4
   - Check if validation loss is decreasing

3. **Import errors**: 
   ```bash
   pip install --upgrade trl peft transformers
   ```

4. **Quality issues**: 
   - Increase LoRA rank
   - Train for more epochs
   - Check data quality and format

5. **Dataset path errors**:
   - Ensure paths in config file are correct
   - Check image folder exists and contains images
   - Verify JSON format is correct

### Memory Optimization

For limited GPU memory:
```json
{
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": true,
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": false
}
```

### Training Monitoring

The script automatically:
- Calculates training steps based on dataset size
- Saves checkpoints at regular intervals
- Provides progress updates
- Monitors GPU usage

Check logs for:
- Training/validation loss curves
- GPU memory usage
- Training speed (samples/sec)

## Advanced Usage

### Custom Target Modules
```json
{
    "lora_target_modules": "q_proj,v_proj,o_proj",
    "apply_lora_to_vision": true
}
```

### Quantized Training (QLoRA)
For even lower memory usage, modify the model loading code to use 4-bit quantization.

## Comparison with Full Fine-tuning

| Method | GPU Memory | Training Speed | Model Quality | Deployment |
|--------|------------|----------------|---------------|------------|
| Full Fine-tuning | 40GB+ | Slow | Best | Simple |
| LoRA | 16-20GB | Fast | Very Good | Flexible |
| QLoRA | 8-12GB | Medium | Good | Complex |

**Recommendation**: Use LoRA for most applications - it provides excellent quality with much better efficiency and flexibility.