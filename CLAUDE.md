# CLAUDE.md

This file provides guidance to Claude Code when working with this VLM research repository.

## Repository Overview

Vision-Language Models (VLMs) research repository with implementations and training pipelines:

- **Ovis2.5** - Open VISion model with tool-calling capabilities (Crop, Identify)
- **InternVL3.5** - International Vision-Language model with RefCOCO grounding
- **Utils** - Dataset generation utilities (RefCOCO, Identity tool-calling datasets)

## Key Architecture Notes

### Ovis2.5
- Uses `AutoModelForCausalLM` (not `AutoModelForVision2Seq`)
- Custom preprocessing: `model.preprocess_inputs(messages)`
- Thinking mode with `<think>` tags for chain-of-thought reasoning
- Tool system with auto-detection from `*_tool.py` files

### InternVL3.5
- Standard HuggingFace Vision2Seq architecture
- Dynamic resolution support
- LoRA fine-tuning compatible

---

## Training Commands

### Ovis2.5 Training

```bash
# SFT (Supervised Fine-Tuning) - Start here
bash Ovis/src_theo/sft/train_launch.sh Ovis/src_theo/sft/train_config.json 4

# LoRA (Memory-efficient)
bash Ovis/src_theo/lora/train_launch_lora.sh

# GRPO (Refinement after SFT - optional)
bash Ovis/src_theo/grpo/train_launch_grpo.sh
```

### InternVL3 Training

```bash
# LoRA fine-tuning
cd InternVL3/src_theo/lora
bash train_launch_lora.sh
```

---

## Inference Commands

### Ovis2.5

```bash
# Web UI
python Ovis/ovis/serve/web_ui.py --model-path AIDC-AI/Ovis2.5-9B --port 8001

# vLLM server
vllm serve AIDC-AI/Ovis2.5-9B --trust-remote-code --port 8000

# Basic inference
python Ovis/ovis/serve/infer_basic_demo.py

# Thinking mode
python Ovis/ovis/serve/infer_think_demo.py
```

### InternVL3

```bash
# Batch inference
python InternVL3/inference_img_batch.py

# Video inference
python InternVL3/inference_vd.py
```

---

## Data Formats

### Ovis Training Data

**Standard conversation:**
```json
{
  "image": "train2017/000000123.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat's in this image?"},
    {"from": "gpt", "value": "This image shows..."}
  ]
}
```

**Tool-calling format:**
```json
{
  "image": "train2017/000000456.jpg",
  "conversations": [
    {"from": "system", "value": "You have access to Crop and Identify tools..."},
    {"from": "human", "value": "<image>\nWho is this person?"},
    {"from": "gpt", "value": "<think>\nLet me identify them.\n<tool_call>Identify [10,20,100,200]</tool_call><tool_response>John Doe</tool_response>\n</think>\n<answer>This is John Doe.</answer>"}
  ]
}
```

**Important:** `<tool_response>` is **masked from loss** during training to prevent hallucination.

### InternVL3 Training Data

```json
[
  {
    "id": "sample_001",
    "image": "image.png",
    "conversations": [
      {"from": "human", "value": "<image>\nQuestion"},
      {"from": "gpt", "value": "Response"}
    ]
  }
]
```

---

## Tool System (Ovis Only)

### Creating Tools

Tools are auto-detected from `Ovis/src_theo/tools/*_tool.py`:

```python
# Ovis/src_theo/tools/your_tool.py
from .tool_base import ToolBase

class YourTool(ToolBase):
    """Tool description for LLM guidance."""

    def extract_tool_call(self, text):
        # Parse single call (inference)
        pass

    def execute(self, image, parameters):
        # Execute tool - return {"type": "image"|"text", "content": ...}
        pass
```

**No configuration needed** - tools are auto-registered!

### Tool Usage

- **Image-returning tools** (Crop): `<tool_call>Crop [x,y,x2,y2]</tool_call><image>`
- **Text-returning tools** (Identify): `<tool_call>Identify [x,y,x2,y2]</tool_call><tool_response>Name</tool_response>`

See `Ovis/src_theo/tools/README_TOOL_SYSTEM.md` for details.

---

## Dataset Generation

### RefCOCO Crop Tool-Calling Dataset

Located in `InternVL3/refcoco/`:

```bash
# Pipeline: merge → generate Q&A → LLM refine → convert → filter
python merge_refcoco_datasets.py
python refcoco_main3.py
python refcoco_main_llm.py
python create_jsonl.py
python convert_QnA_data_to_standard.py
```

**Output:** High-quality crop tool-calling dataset (90.6/100 quality score)

### Identity Tool-Calling Dataset

Located in `InternVL3/refcoco_id/`:

```bash
# Two-stage generation: raw Q&A → refinement
python identity_stage1.py --merged_data merged_refcoco_data.pkl
python identity_stage2.py --stage1_folder refcoco_identity_stage1
```

**Output:** 45k Q&A pairs (31k single-person + 14k multi-person)

See respective README.md files for detailed pipelines.

---

## File Structure

```
VLMs/
├── Ovis/
│   ├── src_theo/
│   │   ├── sft/              # Standard SFT training
│   │   ├── lora/             # LoRA training
│   │   ├── grpo/             # GRPO refinement
│   │   ├── tools/            # Tool system (Crop, Identify)
│   │   └── inference_ovis25.py
│   └── ovis/serve/           # Official inference scripts
│
├── InternVL3/
│   ├── src_theo/lora/        # LoRA training
│   ├── refcoco/              # Crop tool dataset generation
│   └── refcoco_id/           # Identity tool dataset generation
│
└── utils/
    ├── split_dataset.py      # Dataset splitting utility
    └── lh-poc/model/         # LH-POC model files
```

---

## Common Tasks

### Training Ovis2.5 on Tool-Calling Data

1. **Prepare data:** Merge RefCOCO + Identity datasets
2. **Train SFT:** `bash Ovis/src_theo/sft/train_launch.sh ...`
3. **Optional GRPO:** `bash Ovis/src_theo/grpo/train_launch_grpo.sh`

### Adding New Tools

1. Create `Ovis/src_theo/tools/new_tool.py` inheriting from `ToolBase`
2. Implement `extract_tool_call()` and `execute()`
3. Tool auto-registers - no config needed!

### Testing Tool System

```bash
python Ovis/src_theo/tools/test_tool_system.py
```

---

## Important Notes

- **Ovis2.5 batch_size=1** is mandatory (NaViT architecture)
- **Tool responses must be masked** during training (see `conversation_dataset.py`)
- **Use BF16** (`bf16: true`) for better stability than FP16
- **Flash attention** recommended: `pip install flash-attn --no-build-isolation`
- **DeepSpeed ZeRO-2** recommended for Ovis2.5-9B training

---

## References

- **Ovis paper:** NaViT-based multimodal architecture
- **Tool system:** `Ovis/src_theo/tools/README_TOOL_SYSTEM.md`
- **GRPO training:** `Ovis/src_theo/grpo/GRPO_TRAINING_README.md`
- **RefCOCO dataset:** `InternVL3/refcoco/README.md`
- **Identity dataset:** `InternVL3/refcoco_id/README.md`
