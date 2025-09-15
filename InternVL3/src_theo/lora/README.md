# InternVL 3.5 LoRA Fine-tuning

LoRA fine-tuning for InternVL 3.5 models with multimodal and thinking mode support.

## Requirements

```bash
pip install torch torchvision transformers>=4.52.1 trl peft datasets pillow tensorboard
pip install flash-attn  # optional but recommended
```

## Dataset Format

JSON file with conversation format:

```json
[
  {
    "id": "sample_001",
    "image": "image.png",  // optional
    "conversations": [
      {"from": "human", "value": "<image>\nWhat do you see?"},
      {"from": "gpt", "value": "I can see..."}
    ]
  }
]
```

Supports both regular responses and thinking mode with `<think>` tags.

## Dataset Usage

```python
from internvl_dataset import create_internvl_dataset, MultimodalDataCollator

# Create dataset
train_dataset = create_internvl_dataset(
    data_path="/path/to/train_data.json",
    image_folder="/path/to/images/",
    tokenizer=tokenizer
)

# Create data collator  
data_collator = MultimodalDataCollator(tokenizer)
```

## Configuration

Edit `train_config.json`:

```json
{
    "model_name_or_path": "OpenGVLab/InternVL3_5-8B",
    "data_path": "/path/to/train_data.json",
    "image_folder": "/path/to/images/",
    
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_target_modules": [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "mlp1.1",
        "mlp1.3",
        "qkv",
        "fc1",
        "fc2"
    ],
    
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4
}
```

## Training

```bash
# Single GPU
python train_theo_lora.py train_config.json

# Multi-GPU  
bash run_training.sh
```

## Key Parameters

- **`lora_r`**: LoRA rank (32-64 recommended)
- **`lora_alpha`**: Usually 2x rank  
- **`per_device_train_batch_size`**: Adjust based on VRAM
- **`gradient_accumulation_steps`**: For effective batch size

## Monitor Training

```bash
tensorboard --logdir /workspace/VLMs/InternVL3/logs/internvl35_lora_5k_samples
```

## Testing

```bash
python test_inference.py
```

## Troubleshooting

- **CUDA OOM**: Reduce batch size or enable gradient checkpointing
- **Import errors**: Check PYTHONPATH and InternVL installation  
- **Image loading**: Verify paths and formats (PNG/JPG)

## Files

- `train_theo_lora.py`: Main training script
- `internvl_dataset.py`: Reusable dataset module for multimodal training
- `train_config.json`: Training configuration
- `run_training.sh`: Multi-GPU training script
- `test_inference.py`: Inference testing
