# Ovis Custom Training Setup - src_theo

This folder contains a custom training implementation for Ovis models based on the original Ovis training structure. It allows you to train Ovis models with your own datasets.

## Files Overview

- `arguments.py` - Training arguments and configuration classes
- `dataset_theo.py` - Custom dataset class for multimodal data
- `train_theo.py` - Main training script
- `train_config.sh` - Configuration script for easy training setup
- `data_example.json` - Example data format
- `data_utils.py` - Utilities for data preparation and validation

## Prerequisites

1. Install the original Ovis package:
```bash
cd Ovis
pip install -r requirements.txt
pip install -e .
```

2. Make sure you have the required dependencies:
```bash
pip install torch transformers datasets pillow
```

## Data Format

Your training data should follow this JSON structure:

```json
{
  "conversations": [
    {
      "id": "unique_sample_id",
      "image": "path/to/image.jpg",  
      "conversations": [
        {
          "from": "human",
          "value": "<image>\nWhat do you see?"
        },
        {
          "from": "gpt", 
          "value": "I see a cat sitting on a table."
        }
      ]
    }
  ]
}
```

### Special Token Support
The dataset supports Ovis special tokens in conversations:
- **Thinking tokens**: `<think>...</think>` for chain-of-thought reasoning
- **Tool call tokens**: `<tool_call>...</tool_call>` for tool usage
- **Vision tokens**: Automatically handled by Ovis preprocessing
- **Image/Video tokens**: `<image>` and `<video>` placeholders

### Key Points:
- Use `<image>` token to indicate where the image should be processed
- Support for multi-turn conversations with special tokens
- Images can be relative paths (relative to `image_folder`) or absolute paths
- Video support: provide list of frame paths in `"video"` field instead of `"image"`
- Special tokens in gpt responses are automatically tokenized with correct IDs

## Quick Start

### 1. Prepare Your Data

```bash
# Create sample data for testing
python data_utils.py sample ./data/sample_data.json --num_samples 20

# Validate your data format
python data_utils.py validate ./data/train_data.json ./data/images

# Split data into train/val/test sets
python data_utils.py split ./data/full_data.json --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### 2. Configure Training

Edit `train_config.sh` to set your paths and parameters:

```bash
# Data paths
DATA_PATH="./data/train_data.json"
IMAGE_FOLDER="./data/images"

# Model configuration (original Ovis structure)
LLM_MODEL="Qwen/Qwen3-8B"  # or "microsoft/DialoGPT-medium"
VIT_MODEL="google/siglip2-so400m-patch16-512"

# Training parameters (original defaults)
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MULTIMODAL_MAX_LENGTH=4096  # Original default
TEXT_MAX_LENGTH=4096

# Image processing (original Ovis defaults)
SINGLE_IMAGE_MIN_PIXELS=200704  # 448*448
SINGLE_IMAGE_MAX_PIXELS=2408448 # 1792*1344
MULTIPLE_IMAGE_MIN_PIXELS=200704 # 448*448  
MULTIPLE_IMAGE_MAX_PIXELS=802816 # 896*896 (corrected)
VIDEO_MAX_PIXELS=802816 # 896*896 (corrected)
```

### 3. Start Training

```bash
# Make script executable and run
chmod +x train_config.sh
./train_config.sh
```

Or run directly with Python:

```bash
python train_theo.py \
    --llm_name_or_path "Qwen/Qwen3-8B" \
    --vit_name_or_path "google/siglip2-so400m-patch16-512" \
    --visual_vocab_size 65536 \
    --data_path "./data/train_data.json" \
    --image_folder "./data/images" \
    --train_modules "all" \
    --output_dir "./checkpoints/my_ovis_model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --multimodal_max_length 4096 \
    --single_image_min_pixels 200704 \
    --single_image_max_pixels 2408448 \
    --multiple_image_max_pixels 802816 \
    --video_max_pixels 802816
```

## Advanced Configuration

### Model Arguments (following original Ovis structure)

- `llm_name_or_path`: Base language model (e.g., "Qwen/Qwen3-8B")
- `vit_name_or_path`: Vision transformer model path
- `visual_vocab_size`: Size of visual vocabulary (default: 65536)
- `conversation_formatter_class`: Conversation formatting class
- `attn_implementation`: Attention implementation type
- `accepts_loss_kwargs`: Whether model accepts loss kwargs
- `vit_hidden_stride`: ViT hidden stride (default: 2)
- `vit_window_size`: ViT window size (default: 112)
- `vit_temporal_patch_size`: Temporal patch size (default: 1)
- `vit_preserve_original_pe`: Preserve original positional embeddings
- `vit_use_rope`: Use RoPE in ViT

### Training Arguments (original Ovis parameters)

- `train_modules`: Which modules to train (options: "all", "llm", "visual_tokenizer", "visual_tokenizer.head", "visual_tokenizer.vit", "vte")
- `stage`: Training stage number (affects save format: stage < 3 uses .bin, stage >= 3 uses .safetensors)
- `multimodal_max_length`: Maximum multimodal sequence length (default: 4096)
- `text_max_length`: Maximum text sequence length (default: 4096)
- `single_image_min_pixels`: Minimum pixels for single image (default: 448*448 = 200704)
- `single_image_max_pixels`: Maximum pixels for single image (default: 1792*1344 = 2408448)
- `multiple_image_min_pixels`: Minimum pixels for multiple images (default: 448*448 = 200704)
- `multiple_image_max_pixels`: Maximum pixels for multiple images (default: 896*896 = 802816)
- `video_min_pixels`: Minimum pixels for video frames (default: 448*448 = 200704)
- `video_max_pixels`: Maximum pixels for video frames (default: 896*896 = 802816)
- `monitor_step`: Steps between monitoring (default: 100)
- `model_init_seed`: Model initialization seed (default: 0)
- `data_type`: Type of data ("conversation" or "caption")

## Directory Structure

```
Ovis/src_theo/
├── arguments.py           # Training arguments
├── dataset_theo.py        # Custom dataset class
├── train_theo.py          # Main training script
├── train_config.sh        # Training configuration
├── data_utils.py          # Data utilities
├── data_example.json      # Example data format
└── README.md             # This file

data/
├── train_data.json       # Training data
├── eval_data.json        # Evaluation data (optional)
└── images/               # Image files
    ├── image1.jpg
    ├── image2.jpg
    └── ...

checkpoints/
└── my_ovis_model/        # Training outputs
    ├── pytorch_model.bin
    ├── config.json
    ├── training_args.json
    └── logs/
```

## Training Tips

1. **Memory Management**: Use smaller batch sizes and gradient accumulation for limited GPU memory:
   ```bash
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 16
   ```

2. **Training Modules**: Choose what to train based on your needs:
   - Full fine-tuning: `--train_modules "all"`
   - Visual adapter only: `--train_modules "visual_tokenizer.head|vte"`
   - ViT fine-tuning: `--train_modules "visual_tokenizer.vit"`
   - LLM fine-tuning: `--train_modules "llm"`

3. **Learning Rates**: Different modules can have different learning rates:
   ```bash
   --train_modules "visual_tokenizer.head:1e-4|llm:2e-5"
   ```

4. **Image Processing**: Adjust pixel limits based on your GPU memory and content type:
   - Single image high resolution: `--single_image_max_pixels 2408448` (1792*1344)
   - Multiple images/video: `--multiple_image_max_pixels 802816` (896*896) 
   - Low resolution fallback: `--single_image_max_pixels 200704` (448*448)

5. **Special Tokens**: The dataset automatically supports Ovis special tokens:
   - Thinking mode: `<think>reasoning process</think>`
   - Tool calls: `<tool_call>function(args)</tool_call>`
   - Vision markers: Handled automatically by Ovis

6. **Mixed Precision**: Use `--bf16` or `--fp16` to reduce memory usage

6. **Monitoring**: Use `--report_to tensorboard` and check logs:
   ```bash
   tensorboard --logdir ./checkpoints/my_ovis_model/logs
   ```

7. **Resuming Training**: The script automatically resumes from the latest checkpoint in output_dir

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Missing images**: Use `data_utils.py validate` to check data integrity
3. **Import errors**: Make sure Ovis package is properly installed

### Data Validation

Always validate your data before training:

```bash
python data_utils.py validate ./data/train_data.json ./data/images
```

### Performance Monitoring

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./checkpoints/my_ovis_model/logs
```

## Model Usage After Training

After training, you can use your model like this:

```python
from transformers import AutoModelForCausalLM
from PIL import Image

# Load your trained model
model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/my_ovis_model",
    trust_remote_code=True
)

# Use for inference
image = Image.open("test_image.jpg")
text = "What do you see in this image?"
query = f"<image>\n{text}"

prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
# ... continue with generation
```

## Citation

If you use this training code, please cite the original Ovis paper:

```bibtex
@article{lu2024ovis,
  title={Ovis: Structural Embedding Alignment for Multimodal Large Language Model},
  author={Shiyin Lu and Yang Li and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang and Han-Jia Ye},
  year={2024},
  journal={arXiv:2405.20797}
}
```