# InternVL3 Dataset Module Usage

## Overview

The dataset functionality has been refactored into a separate, reusable module: `internvl_dataset.py`

## Files

- **`internvl_dataset.py`** - Standalone dataset module for InternVL3 multimodal training
- **`train_theo_lora.py`** - Main LoRA fine-tuning script (now imports from dataset module)

## Components in `internvl_dataset.py`

### Classes

- **`InternVLDataset`** - Main dataset class for handling multimodal conversations with thinking mode support
- **`MultimodalDataCollator`** - Custom collator for batching text and images

### Functions

- **`create_internvl_dataset()`** - Convenience function to create dataset and convert to HuggingFace format

## Usage Examples

### Basic Usage (LoRA Training)
```python
from internvl_dataset import create_internvl_dataset, MultimodalDataCollator

# Create dataset
train_dataset = create_internvl_dataset(
    data_path="/path/to/train_data.json",
    image_folder="/path/to/images/",
    tokenizer=tokenizer,
    image_size=448,
    max_dynamic_patches=12
)

# Create data collator
data_collator = MultimodalDataCollator(tokenizer)
```

### Full Fine-tuning Usage
```python
from internvl_dataset import InternVLDataset, MultimodalDataCollator

# Direct dataset usage
dataset = InternVLDataset(
    data_path="/path/to/train_data.json",
    image_folder="/path/to/images/",
    tokenizer=tokenizer,
    image_size=448,
    max_dynamic_patches=12
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
data_collator = MultimodalDataCollator(tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)
```

## Benefits

1. **Reusability** - Can be used across different training scripts (LoRA, full fine-tuning, evaluation)
2. **Maintainability** - Centralized dataset logic in one module
3. **Flexibility** - Easy to modify dataset behavior without touching training scripts
4. **Testing** - Can be tested independently from training logic
