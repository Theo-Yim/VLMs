# House Inspection VLM Dataset & Training

Dataset generation pipeline for training Vision-Language Models (VLMs) on house inspection tasks using knowledge distillation from larger VLMs.

## Overview

Creates conversational reasoning datasets by:
1. Using large VLMs to generate detailed reasoning and answers about house defects
2. Refining and formatting data for VLM training
3. Providing WebUI for model demonstration

## Structure

```
lh-poc/
├── dataloader.py                        # Data loading utilities
├── model/
│   ├── run_inference_theo_pa.sh         # Parallel GPU inference launcher
│   ├── inference_theo_pa.py             # Parallel inference worker
│   ├── inference_theo.py                # Single GPU inference
│   ├── prompt_theo.py                   # R1/R2-style prompts (active)
│   ├── prompt_sb_v2.py                  # Alternative prompts (active)
│   ├── train_dataset_preparation.py     # Format converter (theo)
│   ├── train_dataset_preparation_sb.py  # Format converter (sb)
│   ├── name.py                          # Korean→English mappings
│   ├── dataloader_lh_wrapper.py         # PyTorch DataLoader wrapper
│   ├── InternVL3_preprocess.py          # Image preprocessing for InternVL
│   ├── InternVL3_processor_utils.py     # Model loading utilities for InternVL
│   └── rerun_failed_inference.py        # Retry logic for failures. Used in train_dataset_preparation*.py
└── webui/                                # Model demonstration UI
```

## Pipeline

Before starting, PYTHONPATH should be registered.
```bash
export PYTHONPATH="/(path to lh-poc)/lh-poc:$PYTHONPATH"
```

### 1. Generate Reasoning Data
**Multi-GPU (recommended):**
```bash
cd lh-poc/model
./run_inference_theo_pa.sh
```
- Spawns N parallel workers via `inference_theo_pa.py`
- Output: `results_theo_parallel/process_*/`

**Single GPU:**
```bash
python inference_theo.py --data_root /path/to/data --gpu_id 0
```

### 2. Convert to Training Format
**For prompt_theo.py:**
```bash
python train_dataset_preparation.py \
  --base_path results_theo_parallel \
  --output_path dataset_lh_theo_train.json
```

**For prompt_sb_v2.py:**
```bash
python train_dataset_preparation_sb.py \
  --base_path results_sb_parallel \
  --output_path dataset_lh_sb_train.json
```

Converters handle:
- Parsing `<think>` / `</think>` tags
- JSON validation & retry on failure
- Conversation format generation

## Key Components

### Prompts (Active)
**prompt_theo.py** - R1/R2 style:
- `prompt_theo_v2_system`: System message
- `prompt_theo_v2`: Main prompt with `<T>` tags
- Outputs thinking + JSON answer

**prompt_sb_v2.py** - Alternative:
- `R1_SYSTEM_PROMPT`: CoT reasoning protocol
- `ENGLISH_TRAIN_PROMPT`: Ground truth guided
- Outputs structured JSON

### Label Mapping
**name.py** - Korean→English dictionaries:
- `SPACE_CLASS`: 96 types
- `MATERIAL_CLASS`: 412 types
- `DEFECT_CLASS`: 141 types
- `REPAIR_WORK_CLASS`: 45 types

Ensures VLM operates in English for better performance.

### Error Handling
**rerun_failed_inference.py**:
- Auto-retry mechanism for failed inferences
- Used by dataset preparation scripts
- Max 3 retries per item

## Output Format

**Inference** (per image):
```
{label_id}.txt containing:
<think>Step-by-step reasoning...</think>
{"space": "Kitchen", "defect_present": "Yes", ...}
```

**Training-ready dataset**:
```json
{
  "id": "sample_00001",
  "image": "uuid.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n{prompt}"},
    {"from": "gpt", "value": "<think>...</think>\n{json}"}
  ]
}
```