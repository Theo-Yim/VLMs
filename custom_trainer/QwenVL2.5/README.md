# Qwen 2.5 VL Training and Inference

This repository contains a complete implementation for training and inference with Qwen 2.5 Vision-Language Model (VL), featuring two-stage training with Supervised Fine-Tuning (SFT) and Regional Group Preference Optimization (R-GRPO).

## Features

- **Two-Stage Training Pipeline**:
  - Stage 1: Supervised Fine-Tuning (SFT)
  - Stage 2: Regional GRPO (R-GRPO) based on VLM-R3 paper
- **Tool Call Support**: Parse and handle tool calls for regional visual grounding
- **LoRA Support**: Efficient training with Low-Rank Adaptation
- **Flexible Configuration**: Modular config system for easy experimentation
- **Comprehensive Inference**: Full inference pipeline with output parsing

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd QwenVL2.5

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash attention for better performance
pip install flash-attn --no-build-isolation
```

## Data Format

The training data should be in JSONL format. Each line contains a JSON object with the following structure:

```json
{
  "image_path": "coco/train2017/000000580837.jpg",
  "image_id": "000000580837",
  "QnA": [
    {
      "Q": "What is the visible clothing of the person on the far left?",
      "A3": "<think>\nTo determine the visible clothing...\n{Crop person 1 [0.00, 141.43, 79.23, 480.00]}\n</think>\n\n<answer>The visible clothing of the person on the far left is a dark jacket.</answer>"
    }
  ]
}
```

**Key Points:**
- Only `image_path` and `QnA` fields are used for training
- Tool calls in format `{Crop ... [x1, y1, x2, y2]}` are converted to `<tool_call>[x1, y1, x2, y2]</tool_call>`
- After each `</tool_call>`, the cropped image region is fed to the model for continued generation
- **Multiple tool calls in single response**: Each tool call gets its own cropped image inserted immediately after the `</tool_call>` token
- The model learns to use crops for detailed visual analysis with full multi-modal reasoning

**Example with multiple crops:**
```
Input: "{Crop area1 [x1,y1,x2,y2]} text {Crop area2 [a,b,c,d]} more text"

Becomes: 
[Text] + [<tool_call>[x1,y1,x2,y2]</tool_call>] + [Cropped Image 1] + 
[text] + [<tool_call>[a,b,c,d]</tool_call>] + [Cropped Image 2] + [more text]
```

## Training

### Quick Start - Optimal Configuration

Get optimal settings for your dataset size and hardware:

```bash
# Get recommended configuration for your setup
python configure_training.py \
    --dataset_size 40000 \
    --gpu_memory 32 \
    --show_command

# This will output the optimal training command for your case
# Example output for 40k images with 32GB GPU:
python train_qwenvl_25.py \
    --stage both \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --image_base_path data/images \
    --use_lora \
    --lora_r 128 \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --bf16
```

### Manual Configuration

Train both stages with default settings:

```bash
python train_qwenvl_25.py \
    --stage both \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --image_base_path data/images \
    --use_lora \
    --bf16
```

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
python train_sft.py
```

Or with custom parameters:

```bash
python train_qwenvl_25.py \
    --stage sft \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --image_base_path data/images \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --use_lora \
    --lora_r 64 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --sft_output_dir outputs/sft \
    --bf16
```

### Stage 2: Regional GRPO (R-GRPO)

```bash
python train_grpo.py
```

Or continue from SFT checkpoint:

```bash
python train_qwenvl_25.py \
    --stage grpo \
    --train_data data/train.jsonl \
    --image_base_path data/images \
    --sft_checkpoint outputs/sft/final_model \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --grpo_output_dir outputs/grpo \
    --bf16
```

### Using Configuration Files

Create JSON configuration files for more complex setups:

```bash
python train_qwenvl_25.py \
    --stage both \
    --model_config configs/model_config.json \
    --data_config configs/data_config.json \
    --sft_config configs/sft_config.json \
    --grpo_config configs/grpo_config.json
```

Example configuration file (`configs/sft_config.json`):

```json
{
  "output_dir": "outputs/sft",
  "learning_rate": 2e-5,
  "num_train_epochs": 3,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "gradient_accumulation_steps": 4,
  "bf16": true,
  "use_lora": true,
  "lora_r": 64,
  "lora_alpha": 128,
  "lora_dropout": 0.1
}
```

## Inference

### Command Line Interface

```bash
python inference.py \
    --model_path outputs/grpo/final_model \
    --image_path test_image.jpg \
    --question "What objects can you see in this image?" \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --output_json output.json
```

### Python API

```python
from inference import QwenVLInference
from config import InferenceConfig
from PIL import Image

# Initialize inference
config = InferenceConfig(
    model_path="outputs/grpo/final_model",
    max_new_tokens=512,
    temperature=0.7,
    parse_tool_calls=True
)
inference = QwenVLInference(config)

# Single image inference
image = Image.open("test_image.jpg")
result = inference.generate(
    image=image,
    question="What objects can you see in this image?"
)

# Access parsed output
print("Thinking:", result.think_content)
print("Answer:", result.answer_content)
print("Tool Calls:", result.tool_calls)

# Batch inference
images = [Image.open(f"image_{i}.jpg") for i in range(3)]
questions = ["What is this?", "Describe the scene", "Count the objects"]
results = inference.batch_generate(images, questions)
```

## Custom Tokens

The model uses custom tokens for tool calling:
- `<tool_call>` (ID: 151657): Start of tool call
- `</tool_call>` (ID: 151658): End of tool call

These are automatically added during training and handled during inference.

## Key Features Explained

### Crop Tool Functionality

The crop tool is central to the model's ability to perform detailed visual analysis:

1. **Training Phase:**
   - Tool calls in format `{Crop description [x1, y1, x2, y2]}` are parsed
   - Converted to `<tool_call>[x1, y1, x2, y2]</tool_call>` tokens
   - The specified region is cropped from the image
   - The cropped image is fed to the model immediately after the `</tool_call>` token
   - The model continues generation with access to both the full image and the crop

2. **Inference Phase:**
   - Model generates text with `<tool_call>` tokens
   - System automatically executes crops
   - Cropped regions can be fed back for multi-turn interaction
   - Enables focused analysis of specific image regions

### Regional GRPO

The R-GRPO stage implements regional reward computation:
- Validates tool call coordinates
- Assesses structural quality of responses (think/answer format)
- Rewards appropriate use of crop tools
- Balances global and regional understanding

### Image Processing

**Main Images**: Resized so the shorter side equals 448 pixels while maintaining aspect ratio.

**Cropped Regions**: Use smart resizing that:
- **Preserves original size** if shorter side ≤ 448px (avoids upscaling small crops)
- **Downsizes only when needed** if shorter side > 448px (prevents memory issues)
- **Maintains aspect ratio** in all cases

This approach preserves the quality of small cropped regions while controlling maximum sizes for memory efficiency.

## Project Structure

```
QwenVL2.5/
├── train_qwenvl_25.py      # Main training orchestrator
├── train_sft.py            # Stage 1: SFT training
├── train_grpo.py           # Stage 2: R-GRPO training
├── inference.py            # Inference with tool call parsing
├── configure_training.py   # Training configuration tool
├── data_utils.py           # Data loading and preprocessing
├── config.py               # Configuration dataclasses
├── example_usage.py        # Usage examples and demos
├── requirements.txt        # Python dependencies
└── README.md              # This file
crop_tool.py                # Crop tool implementation
```

## Training Tips

### LoRA vs Full Fine-tuning Decision

**Use LoRA (Recommended for most cases):**
- Dataset < 100k images
- Limited GPU memory (< 40GB VRAM)
- Initial experimentation phase
- Consumer GPU setups

**Consider Full Fine-tuning when:**
- Dataset > 100k images
- LoRA results plateau or underperform
- Have access to high-memory GPUs (80GB+)
- Production deployment requiring maximum performance

### Configuration Guidelines by Dataset Size:

**Small datasets (< 10k images):**
```bash
--use_lora --lora_r 64 --learning_rate 2e-5
```

**Medium datasets (10k-50k images) - Your case:**
```bash
--use_lora --lora_r 128 --learning_rate 2e-5
```

**Large datasets (50k-100k images):**
```bash
--use_lora --lora_r 256 --learning_rate 1e-5
# OR full fine-tuning: --learning_rate 5e-6 (without --use_lora)
```

**Very large datasets (> 100k images):**
```bash
# Full fine-tuning recommended
--learning_rate 5e-6  # No LoRA flags
```

### General Tips:

1. **Start with LoRA**: Use optimized config above based on your dataset size
2. **Gradient Accumulation**: Increase `--gradient_accumulation_steps` if batch size is limited by GPU memory
3. **Mixed Precision**: Use `--bf16` for better stability than `--fp16`
4. **Learning Rate**: Start with 2e-5 for LoRA SFT, 5e-6 for full fine-tuning
5. **Validation**: Monitor validation loss to prevent overfitting
6. **Tool Calls**: Ensure your data includes diverse tool call examples for better grounding
7. **Memory Optimization**: For LoRA with large datasets, consider rank 128-256 for better capacity

## Troubleshooting

### Out of Memory
- Reduce batch size
- Increase gradient accumulation steps
- Use LoRA with smaller rank
- Enable gradient checkpointing

### Slow Training
- Install flash attention: `pip install flash-attn`
- Use bf16 mixed precision training
- Reduce max sequence length if appropriate

### Poor Tool Call Performance
- Ensure training data has diverse and accurate tool call annotations
- Adjust regional reward weights in GRPO config
- Increase training epochs for GRPO stage

## Citation

If you use this code, please cite:

```bibtex
@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}
```

For R-GRPO methodology:

```bibtex
@article{vlm-r3,
    title = {VLM-R3: Regional Reinforcement Learning from Human Feedback},
    url = {https://arxiv.org/abs/2505.16192},
    year = {2024}
}
```

## License

This project follows the Apache 2.0 license, consistent with the Qwen2.5-VL model license.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the configuration options in `config.py`
3. Ensure data format matches the specification
4. Open an issue with detailed error messages and configuration