# Ovis2.5 Custom Training & Inference - src_theo

This folder contains a complete implementation for fine-tuning and inference with Ovis2.5 multimodal models, based on the original Ovis2.5 framework with custom modeling.

## Features

- **Fine-tuning**: Complete training pipeline using original Ovis framework components
- **Inference**: Support for single/multi-image, video, text-only, and grounding tasks
- **Thinking Mode**: Support for reflective reasoning with budget control
- **Custom Implementation**: Uses custom Ovis2.5 modeling with original training framework
- **Original Dataset**: Leverages tested `ConversationDataset` from Ovis framework

## Installation

```bash
# Core dependencies
pip install torch transformer numpy pillow moviepy
pip install flash-attn">=2.7.0.post2" --no-build-isolation

# Additional training dependencies
pip install deepspeed accelerate wandb

# Ensure the Ovis framework is available in your Python path
# The training script imports from ovis.train.dataset
```

**Note**: Make sure the original Ovis framework is properly installed and accessible, as the training script uses `ConversationDataset` and `DataCollatorForMultimodalDataset` from the original implementation.

## Quick Start

### 0. Default Parameters

```bash
# Model configuration (original Ovis structure)
LLM_MODEL="Qwen/Qwen3-8B"  # or "microsoft/DialoGPT-medium"
VIT_MODEL="google/siglip2-so400m-patch16-512"
OVIS25_MODEL="AIDC-AI/Ovis2.5-9B"

# Training parameters (original defaults)
MULTIMODAL_MAX_LENGTH=8192  # Original default
TEXT_MAX_LENGTH=4096

# Inference parameters - Thinking mode & budget
enable_thinking = True  # either True or False
enable_thinking_budget = True  # Only effective if enable_thinking is True.
# Inference parameters
max_new_tokens=1024 if enable_thinking is False else 3096
thinking_budget=2048

# Image processing (original Ovis defaults)
SINGLE_IMAGE_MIN_PIXELS=200704  # 448*448
SINGLE_IMAGE_MAX_PIXELS=3211264 # 1792*1792
MULTIPLE_IMAGE_MIN_PIXELS=200704 # 448*448  
MULTIPLE_IMAGE_MAX_PIXELS=802816 # 896*896
VIDEO_MAX_PIXELS=802816 # 896*896
```

### 1. Inference

```python
from src_theo.inference_ovis25 import Ovis25Inference

# Initialize model
ovis = Ovis25Inference(model_path="AIDC-AI/Ovis2.5-9B")

# Single image inference
response = ovis.single_image_inference(
    image_input="path/to/image.jpg",
    text_prompt="Describe this image in detail.",
    enable_thinking=True,
    thinking_budget=2048,
    max_new_tokens=3072
)
print(response)
```

### 2. Fine-tuning

```bash
# Prepare your data in the conversation format (see Data Format section)
# The training uses original Ovis ConversationDataset and DataCollatorForMultimodalDataset
# Configure training parameters in config/train_config.json
# Run training
./run_training.sh config/train_config.json
```

**Note**: This implementation uses the original `ConversationDataset` from `ovis.train.dataset.conversation_dataset` and `DataCollatorForMultimodalDataset` from `ovis.train.dataset.multimodal_dataset` for maximum compatibility and reliability.

## Data Format

### Conversation Dataset Format

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
          "value": "I see a beautiful orange tabby cat sitting on a wooden table..."
        }
      ]
    },
    {
      "id": "sample_002_thinking",
      "image": "math_equation.jpg", 
      "conversations": [
        {
          "from": "human",
          "value": "<image>\nSolve this equation step by step."
        },
        {
          "from": "gpt",
          "value": "<think>\nLet me analyze this equation carefully...\n</think>\n\nTo solve this equation: [solution steps]"
        }
      ]
    },
    {
      "id": "sample_003_multi_image",
      "image": ["image1.jpg", "image2.jpg", "image3.jpg"],
      "conversations": [
        {
          "from": "human", 
          "value": "<image>\n<image>\n<image>\nCompare these images."
        },
        {
          "from": "gpt",
          "value": "Looking at these three images, I can see..."
        }
      ]
    },
    {
      "id": "sample_004_video",
      "video": ["frame1.jpg", "frame2.jpg", "frame3.jpg", "frame4.jpg"],
      "conversations": [
        {
          "from": "human",
          "value": "<video>\nDescribe what happens in this video."
        },
        {
          "from": "gpt", 
          "value": "This video sequence shows..."
        }
      ]
    }
  ]
}
```

## Configuration

### Training Configuration (`src_theo/train_config.json`)

```json
{
  "model_path": "AIDC-AI/Ovis2.5-9B",
  "data_name": "custom_data",
  "data_type": "conversation",
  "data_path": "./data/train_data.json",
  "image_folder": "./data/images",
  "output_dir": "./checkpoints/ovis25_finetune",
  
  "num_train_epochs": 3,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "learning_rate": 2e-5,
  
  "train_modules": "all",
  "freeze_vision_tower": false,
  "freeze_llm": false,
  
  "ovis_pretrained_path": "AIDC-AI/Ovis2.5-9B",
  "stage": 3,
  "multimodal_max_length": 8192,
  "text_max_length": 4096,
  "single_image_max_pixels": 3211264,
  "multiple_image_max_pixels": 802816
}
```

### Key Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `data_name` | Dataset identifier | Any string identifier |
| `data_type` | Data format type | `"conversation"` |
| `train_modules` | Which modules to train | `"all"`, `"llm"`, `"visual_tokenizer"`, `"vte"` |
| `stage` | Training stage | `3` for full training |
| `multimodal_max_length` | Max sequence length with vision | `8192` (recommended) |
| `text_max_length` | Max sequence length text-only | `4096` (recommended) |
| `freeze_vision_tower` | Freeze vision encoder | `true`/`false` |
| `freeze_llm` | Freeze language model | `true`/`false` |
| `*_max_pixels` | Image resolution limits | Adjust based on GPU memory |

## Advanced Usage

### 1. Selective Module Training

```json
{
  "train_modules": "vte",
  "freeze_vision_tower": true,
  "freeze_llm": true
}
```

### 2. Multi-GPU Training

```bash
# Using accelerate
accelerate launch src_theo/train_theo.py config/train_config.json

# Using torchrun  
torchrun --nproc_per_node=4 src_theo/train_theo.py config/train_config.json
```

### 3. DeepSpeed Integration

```json
{
  "deepspeed": "config/deepspeed_config.json",
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16
}
```

### 4. Thinking Mode Training

Include thinking examples in your data:

```json
{
  "from": "gpt",
  "value": "<think>\nLet me think about this step by step...\nFirst, I need to...\nThen, I should...\n</think>\n\nBased on my analysis, the answer is..."
}
```

## Inference Modes

### 1. Standard Inference

```python
response = ovis.single_image_inference(
    image_input="image.jpg",
    text_prompt="Describe this image.",
    max_new_tokens=1024
)
```

### 2. Thinking Mode

```python
response = ovis.single_image_inference(
    image_input="image.jpg", 
    text_prompt="Analyze this complex diagram.",
    enable_thinking=True,
    thinking_budget=2048,
    max_new_tokens=3072
)
```

### 3. Streaming Output

```python
for token in ovis.single_image_inference_streaming(
    image_input="image.jpg",
    text_prompt="Tell me a story about this image.",
    enable_thinking=True
):
    print(token, end='', flush=True)
```

### 4. Visual Grounding

```python
response = ovis.grounding_inference(
    image_input="image.jpg",
    text_prompt="Find the <ref>red apple</ref> in the image.",
    request_type="box"
)
# Output includes: <box>(x1,y1),(x2,y2)</box>
```

## Performance Tips

### Memory Optimization

1. **Reduce image resolution**: Lower `*_max_pixels` values
2. **Gradient checkpointing**: Set `"gradient_checkpointing": true`
3. **Mixed precision**: Use `"bf16": true`
4. **Batch size**: Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`

### Training Speed

1. **Flash Attention**: Use `"attn_implementation": "flash_attention_2"`
2. **Compile**: Set `"torch_compile": true` (experimental)
3. **DataLoader**: Optimize `dataloader_num_workers`

## Directory Structure

```
src_theo/
├── train_theo.py              # Main training script (uses original Ovis components)
├── inference_ovis25.py        # Inference wrapper
├── config/
│   ├── train_config.json      # Training configuration
│   └── deepspeed_config.json  # DeepSpeed configuration
├── data/
│   ├── train_data.json        # Training data
│   └── images/                # Image files
├── checkpoints/               # Model checkpoints
└── run_training.sh           # Training launcher

# Original Ovis components used:
Ovis/ovis/train/dataset/conversation_dataset.py          # ConversationDataset
Ovis/ovis/train/dataset/multimodal_dataset.py           # DataCollatorForMultimodalDataset
Ovis/ovis/train/arguments.py                            # TrainingArguments
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size and image resolution
2. **Slow training**: Enable flash attention and gradient checkpointing
3. **Data loading errors**: Check image paths and formats
4. **Tokenizer issues**: Ensure proper chat template format
5. **Import errors**: Ensure original Ovis framework is in Python path
6. **Dataset format**: Use the exact format expected by original `ConversationDataset`

### Framework Integration

**Using Original Components**: This implementation leverages the battle-tested components from the original Ovis framework:
- `ConversationDataset`: Handles multimodal conversation data loading
- `DataCollatorForMultimodalDataset`: Efficient batch processing for training
- `TrainingArguments`: Extended arguments compatible with Ovis training pipeline

**Benefits**: 
- ✅ Proven reliability and performance
- ✅ Proper handling of complex multimodal data
- ✅ Optimized memory usage and batching
- ✅ Compatible with original training strategies

### Memory Requirements

| Model | Min GPU Memory | Recommended |
|-------|----------------|-------------|
| Ovis2.5-2B | 12GB | 16GB |
| Ovis2.5-9B | 24GB | 32GB |

## Citation

```bibtex
@article{lu2025ovis25technicalreport,
  title={Ovis2.5 Technical Report}, 
  author={Shiyin Lu and Yang Li and Yu Xia and Yuwei Hu and others},
  year={2025},
  journal={arXiv:2508.11737}
}
```