# Ovis2.5-9B Training and Inference

This repository contains a complete implementation for training and inference with Ovis2.5-9B, featuring two-stage training with Supervised Fine-Tuning (SFT) and Regional Group Preference Optimization (R-GRPO).

## Features

- **Two-Stage Training Pipeline**:
  - Stage 1: Supervised Fine-Tuning (SFT)
  - Stage 2: Regional GRPO (R-GRPO) based on VLM-R3 paper
- **Native Resolution Processing**: Ovis2.5 processes images at original resolutions using NaViT
- **Thinking Mode**: Advanced reasoning with thinking budget control
- **Grounding Support**: Built-in support for `<ref>`, `<box>`, `<point>` grounding format
- **LoRA Support**: Efficient training with Low-Rank Adaptation (✅ **Now Working!**)
- **Crop Tool Integration**: Uses the same CropTool as QwenVL2.5

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install flash attention for better performance
pip install flash-attn --no-build-isolation
```

## ⚠️ CRITICAL: Batch Size Requirement

**Ovis2.5 REQUIRES batch_size=1** due to native resolution processing. Different images have variable tensor sizes and cannot be batched.

- ❌ `--batch_size 2` or higher will cause: `RuntimeError: stack expects each tensor to be equal size`
- ✅ Always use batch_size=1 (default) - this is non-negotiable
- 💡 Use `--gradient_accumulation_steps 8-16` to compensate for effective batch size
- 🔧 Training will be slower but memory efficient due to this constraint

## Data Format

The training data should be in JSONL format. Each line contains a JSON object with the following structure:

```json
{
  "image_path": "coco/train2017/000000580837.jpg",
  "QnA": [
    {
      "Q": "What is the visible clothing of the person on the far left?",
      "A3": "<think>\nI need to analyze the person on the far left...\n{Crop person [0.00, 141.43, 79.23, 480.00]}\nLooking at this cropped region, I can see the details.\n</think>\n\n<answer>The person is wearing a dark jacket.</answer>"
    }
  ]
}
```

**Key Points:**
- Tool calls in format `{Crop ... [x1, y1, x2, y2]}` are converted to `<tool_call>[x1, y1, x2, y2]</tool_call>`
- After each `</tool_call>`, the cropped image region is fed to the model for continued generation
- **Multiple tool calls in single response**: Each tool call gets its own cropped image inserted immediately after the `</tool_call>` token
- The model learns to use crops for detailed visual analysis with full multi-modal reasoning

## Training

### Quick Start - Complete Training

```bash
python train_ovis25.py \
    --stage both \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --image_base_path data/images \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 8 \
    --bf16
    # batch_size defaults to 1 (required for native resolution)
```

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
python train_sft.py
```

Or with custom parameters:

```bash
python train_ovis25.py \
    --stage sft \
    --train_data data/train.jsonl \
    --val_data data/val.jsonl \
    --image_base_path data/images \
    --model_name AIDC-AI/Ovis2.5-9B \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --gradient_accumulation_steps 8 \
    --sft_output_dir outputs/sft \
    --bf16
    # batch_size=1 is required and automatic
```

### Stage 2: Regional GRPO (R-GRPO)

```bash
python train_grpo.py
```

Or continue from SFT checkpoint:

```bash
python train_ovis25.py \
    --stage grpo \
    --train_data data/train.jsonl \
    --image_base_path data/images \
    --sft_checkpoint outputs/sft/final_model \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --gradient_accumulation_steps 16 \
    --grpo_output_dir outputs/grpo \
    --bf16
    # batch_size automatically set to 1
```

### Using Configuration Files

Create JSON configuration files for more complex setups:

```bash
python train_ovis25.py \
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
  "gradient_accumulation_steps": 8,
  "bf16": true,
  "use_lora": true,
  "lora_r": 128,
  "lora_alpha": 256,
  "lora_dropout": 0.1
}
```

## LoRA Training

LoRA is **strongly recommended** for Ovis2.5 training. A compatibility patch is required and included:

```bash
# ✅ LoRA patch is applied automatically
python train_ovis25.py \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --learning_rate 2e-5 \
    --bf16
    # batch_size=1 required - cannot be changed
```

**Note**: The `lora_patch.py` file is required in your directory for LoRA training to work. It's included and applied automatically.

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
from inference import OvisInference
from config import InferenceConfig
from PIL import Image

# Initialize
config = InferenceConfig(model_path="outputs/grpo/final_model")
inference = OvisInference(config)

# Generate response
result = inference.generate(
    image=Image.open("test.jpg"),
    question="What do you see in this image?"
)

print("Response:", result['response'])
print("Thinking:", result['parsed']['think_content'])
print("Answer:", result['parsed']['answer_content'])
```

## Architecture Differences from QwenVL2.5

| Component | QwenVL2.5 | Ovis2.5-9B |
|-----------|-----------|------------|
| **Model Loading** | `AutoModelForVision2Seq` | `AutoModelForCausalLM` |
| **Preprocessing** | `AutoProcessor` | `model.preprocess_inputs()` |
| **Tokenizer** | `processor.tokenizer` | `model.text_tokenizer` |
| **Vision Processing** | Fixed/dynamic resolution | Native resolution (NaViT) |
| **Generation** | Standard HuggingFace | Custom with thinking mode |
| **Grounding** | Tool calls format | `<ref>`, `<box>`, `<point>` tags |
| **Batch Size** | Flexible | **Must be 1** (native resolution) |

## Memory Requirements

| Configuration | VRAM Usage | Recommended GPU | Notes |
|---------------|------------|-----------------|-------|
| **Inference** | ~18GB | RTX 4090, A100 | Native resolution processing |
| **LoRA Training (r=128, batch=1)** | ~28GB | RTX 6000 Ada, A100 | Recommended setting |
| **LoRA Training (r=64, batch=1)** | ~24GB | RTX 4090, A100 | If OOM with r=128 |
| **LoRA Training (r=32, batch=1)** | ~20GB | RTX 4090 | Minimum recommended |
| **Full Fine-tuning** | ~40GB+ | A100 80GB | Not recommended |

**Note**: Batch size is always 1 due to native resolution processing. Use `gradient_accumulation_steps` for effective batch size.

## Generation Parameters

Ovis2.5 supports advanced generation with thinking mode:

```python
generation_kwargs = {
    "max_new_tokens": 3072,      # As per official guide
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "enable_thinking": True,      # Enable thinking mode
    "enable_thinking_budget": True,
    "thinking_budget": 2048,     # Thinking tokens budget
}
```

## Troubleshooting

### Common Issues

1. **"RuntimeError: stack expects each tensor to be equal size"**
   - **Cause**: batch_size > 1 
   - **Solution**: Always use batch_size=1 (default) - this is required for native resolution

2. **"AttributeError: no attribute 'get_input_embeddings'"**
   - **Cause**: Missing LoRA compatibility patch
   - **Solution**: Ensure `lora_patch.py` is in your directory (included automatically)

3. **"leaf Variable that requires grad is being used in an in-place operation"**
   - **Cause**: Old version of data_utils.py
   - **Solution**: Update to latest data_utils.py with tensor detachment fixes

4. **"Error processing answer with crop tool: 'type'"**
   - **Cause**: Content validation error in crop tool processing
   - **Solution**: Update to latest data_utils.py with improved error handling

### Performance Tips

1. **Use BF16**: `--bf16` for better stability than FP16
2. **Flash Attention**: Automatically enabled for better performance  
3. **Gradient Checkpointing**: Enabled by default to save memory
4. **Effective Batch Size**: Use `--gradient_accumulation_steps 8-16` since batch_size=1
5. **LoRA Recommended**: Essential for memory efficiency with batch_size=1 constraint

### Memory Optimization

If you encounter CUDA OOM errors:

1. **Increase gradient accumulation**: `--gradient_accumulation_steps 16` (or 32)
2. **Reduce LoRA rank**: `--lora_r 64` (then 32 if still OOM)
3. **Use memory efficient mode**: `--memory_efficient`
4. **Reduce image resolution**: `--max_pixels 448448` (reduce from default 896*896)

## File Structure

```
Ovis2.5/
├── config.py              # Configuration classes
├── train_sft.py           # Stage 1: Supervised Fine-Tuning
├── train_grpo.py          # Stage 2: Regional GRPO
├── train_ovis25.py        # Comprehensive training script
├── data_utils.py          # Data loading and preprocessing
├── inference.py           # Inference implementation
├── lora_patch.py          # LoRA compatibility patch (required)
├── test_ovis_integration.py # Integration tests
├── requirements.txt       # Dependencies
└── README.md             # This file
crop_tool.py               # Crop tool implementation
```

## Key Constraints and Features

### ✅ Advantages
- **Native Resolution**: Preserves fine details and global structure
- **Advanced Reasoning**: Thinking mode with budget control
- **Memory Efficient**: LoRA training works well
- **High Quality**: Better visual understanding than fixed-resolution models

### ⚠️ Constraints
- **Batch Size = 1**: Required due to variable image tensor sizes
- **Slower Training**: Due to batch size constraint
- **Memory Planning**: Need to plan effective batch size via gradient accumulation
- **LoRA Patch Required**: Custom compatibility layer needed

## Citation

If you use this implementation, please cite:

```bibtex
@article{lu2025ovis25technicalreport,
  title={Ovis2.5 Technical Report}, 
  author={Shiyin Lu and Yang Li and Yu Xia and Yuwei Hu and Shanshan Zhao and Yanqing Ma and Zhichao Wei and Yinglun Li and Lunhao Duan and Jianshan Zhao and Yuxuan Han and Haijun Li and Wanying Chen and Junke Tang and Chengkun Hou and Zhixing Du and Tianli Zhou and Wenjie Zhang and Huping Ding and Jiahe Li and Wen Li and Gui Hu and Yiliang Gu and Siran Yang and Jiamang Wang and Hailong Sun and Yibo Wang and Hui Sun and Jinlong Huang and Yuping He and Shengze Shi and Weihong Zhang and Guodong Zheng and Junpeng Jiang and Sensen Gao and Yi-Feng Wu and Sijia Chen and Yuhui Chen and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang},
  year={2025},
  journal={arXiv:2508.11737}
}
```