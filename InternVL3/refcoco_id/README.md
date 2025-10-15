# Identity Tool Calling Dataset Generation

## Overview

**Goal**: Train VLMs to use an **Identify tool** for recognizing people in images through tool-calling. Additionally, it will give the model the ability to understand person-referring in the prompt.

**Dataset**: 45k Q&A pairs (31k single-person + 14k multi-person) with sequential tool calls
- **Single-person Q&A**: "Who is the person wearing blue?" → 1 tool call
- **Multi-person Q&A**: "Who are all the people visible?" → 2-3 tool calls

**Quick Start**:
```bash
./generate_id_dataset_multi.sh  # Generates complete 45k dataset (~8 hours, 3 GPUs)
```

## Purpose

The model should:
1. Observe visual details (clothing, position, actions)
2. Call `<tool_call>Identify [x,y,x2,y2]</tool_call>` with person's bounding box
3. Receive `<tool_response>Name</tool_response>`
4. Answer identity questions using the retrieved name(s)

## Context

**Problem**: Need large-scale tool-calling training data, but manually annotating people's identities is impractical.

**Solution**: Generate synthetic data using RefCOCO person annotations:
- Input: COCO images with person bboxes + rich descriptions (e.g., "woman in blue jacket on the left")
- Output: Conversations with identity questions, tool calls, and mock names

**Inspiration from crop tool-calling dataset** (`InternVL3/refcoco/`):
- The RefCOCO crop dataset successfully taught VLMs to use `{{Crop object_name}}` tool for visual grounding
- This dataset is generated through multiple stages composed of VLM, LLM, and rules
- Architecture: VLM generates Q&A with crop tool calls (through 2-stage; Question generation; Answer generation, using two different prompts) → LLM refines formatting (`refcoco_main3.py` + `refcoco_main_llm.py`)
- Key files to reference:
  - `InternVL3/refcoco/README.md` - Complete guide document for crop tool-calling dataset generation
  - `refcoco_main3.py` - VLM-based Q, A1, A2 generation pipeline
  - `refcoco_main_llm.py` - LLM-based refinement with metrics tracking, generating the final answer A3
  - `../utils/processor_main.py` - Core processor with prompts and tool call handling
  - `../utils/toolcall_parser_n_fixer.py` - Tool call parsing and fixing utilities
- Results:/mnt/nas3/Data/coco/refcoco_vlm_results_theo_llm/000000*.json
- Merged RefCOCO datasets with semantic similarity filtering → `merged_refcoco_data.pkl`, and read this for RefCOCO annotations

**Current approach** (LLM-only, two-stage with enhanced prompts):
1. **Stage 1**: LLM generates raw Q&A with rich reasoning patterns (enhanced prompts with few-shot examples)
2. **Stage 2**: Post-processing refines format + enriches answers with question context

**Note**: Current implementation achieves high quality through improved prompts (see "Current Results & Quality" below). VLM-based approach (Option 2) remains a future enhancement for visual grounding.

## Pipeline

### Stage 1: Raw Generation (`identity_stage1.py`)

**What it does**:
- Loads RefCOCO data filtered to person annotations (~14k images, ~31k people)
- **Generates 1 Q&A per person**: 1 mock name + 1 question + 1 answer with rich reasoning
- Uses enhanced prompts with few-shot examples for diversity
- Generates 2-4 sentences of visual observation before tool calls
- Saves intermediate JSON per image with `{{Identify person #X}}` placeholders

**Output format**:
```json
{
  "image_id": "581857",
  "person_names": ["Omar Hassan", "Elena Volkov"],
  "QnA": [
    {
      "Q": "Who is the woman wearing the blue jacket?",
      "A_raw": "<think>..{{Identify person #1}}..</think>\n<answer>...</answer>",
      "person_num": 1
    }
  ]
}
```

**Usage**:
```bash
CUDA_VISIBLE_DEVICES=1 python identity_stage1.py \
    --merged_data ../../merged_refcoco_data.pkl \
    --coco_path /mnt/nas3/Data/coco \
    --output_folder refcoco_identity_stage1 \
    --start 0 --end 1000
```

**Issues at this stage** (expected, fixed in Stage 2):
- Placeholders `{{Identify person #X}}` instead of actual tool calls (by design)
- Occasional malformed questions (incomplete sentences, missing `?`)
- Minor formatting inconsistencies in `<think>` and `<answer>` tags

**Note**: V2 improvements significantly reduced duplicate names and improved reasoning quality.

### Stage 2: Refinement (`identity_stage2.py`)

**What it does**:
- Loads all Stage 1 JSON files
- For each Q&A pair:
  - Fixes malformed questions (extracts valid question, adds `?`)
  - Validates `<think>` structure (ensures proper reasoning flow)
  - **Enriches answers** with context from questions (e.g., "The person wearing a blue jacket is X")
  - Extracts visual context (clothing, position, actions) and integrates into answers
  - Replaces `{{Identify person #X}}` → `<tool_call>Identify [x,y,x2,y2]</tool_call><tool_response>Name</tool_response>`
  - Ensures proper structure: `<think>...</think>\n<answer>...</answer>`
  - Handles grammar (fixes double articles, preserves original articles from questions)
  - Adds system prompt about Identify tool
- Converts to array of conversation objects (one per Q&A)
- Tracks metrics: `malformed_questions`, `fixed_tool_calls`, `enriched_answers`, etc.

**Output format** (matches `identity_qa_pairs_sample.json`):
```json
[
  {
    "image": "train2017/000000581857.jpg",
    "image_id": "581857",
    "conversations": [
      {"from": "system", "value": "You have access to an Identify tool..."},
      {"from": "human", "value": "<image>\nWho is the person wearing the blue jacket?"},
      {
        "from": "gpt",
        "value": "<think>\nLooking at the image, I can see someone on the left wearing a blue jacket. Let me identify them.\n<tool_call>Identify [103.9,300.0,238.2,477.4]</tool_call><tool_response>Omar Hassan</tool_response>\nThe identification confirms this is Omar Hassan.\n</think>\n<answer>The person wearing the blue jacket is Omar Hassan.</answer>"
      }
    ]
  }
]
```

**Usage**:
```bash
python identity_stage2.py \
    --stage1_folder refcoco_identity_stage1 \
    --output identity_qa_pairs_final.json
```

**Metrics tracked**:
- `malformed_questions`, `fixed_tool_calls`, `enriched_answers`, `missing_think_tags`, `missing_answer_tags`, `removed_qna`

## Key Design Decisions

1. **Why two stages?** Easier debugging - can inspect raw LLM output before refinement
2. **Why 1 name/question per person?** Simple, predictable mapping; prevents combinatorial explosion
3. **Why mock names?** Real identity annotation is expensive; mock names teach tool-calling behavior
4. **Why post-processing (Stage 2)?** LLMs are unreliable at formatting; deterministic fixes ensure quality
5. **Why save per-image JSONs in Stage 1?** Resume capability, easier to debug individual images

## Training Implementation

### Tool Response Masking (CRITICAL)

**Location**: `Ovis/ovis/train/dataset/conversation_dataset.py`

During training, `<tool_response>...</tool_response>` is **masked** (excluded from loss computation) to ensure proper training-inference alignment:

**What gets trained:**
- ✅ `<tool_call>Identify [x,y,x2,y2]</tool_call>` → Model learns to generate tool calls
- ❌ `<tool_response>Name</tool_response>` → **MASKED** (system provides this, not model)
- ✅ Text after `</tool_response>` → Model learns to use returned information

**Why masking is essential:**
- **Without masking**: Model learns to predict/hallucinate names instead of waiting for system
- **With masking**: Model learns tool call → wait for system response → use response
- **Result**: Perfect training-inference alignment

**Verification**: Run `python Ovis/src_theo/tools/test_masking.py` to verify masking works correctly.

**Training-Inference Alignment:**

| Phase | Behavior | Tool Response Source |
|-------|----------|---------------------|
| **Training** | Model generates `<tool_call>`, observes `<tool_response>` (masked), continues reasoning | Training data (masked from loss) |
| **Inference** | Model generates `<tool_call>`, pauses, system inserts `<tool_response>`, model resumes | External identification system |

This ensures model learns to **delegate** identification to external system, not predict names.

## Quality & Status

**✅ Production Ready** - Meets ideal quality standards (see [QUALITY_COMPARISON.md](QUALITY_COMPARISON.md))

**Key features:**
- Rich pre-tool reasoning (2-4 sentences of visual observation before tool call)
- Context-aware answers (e.g., "The person wearing a blue jacket is X")
- Diverse question styles (8 templates, 12-20 words)
- Natural reasoning flow: Observe → locate → tool call → integrate response

**Limitations:**
- Text-based reasoning (LLM relies on RefCOCO descriptions, not actual visual content)
- Mock names only (by design - teaches tool-use pattern, not specific knowledge)

**Future option:** If more visual diversity needed, adapt VLM-based approach from `../refcoco/refcoco_main3.py`

## Files

**Scripts:**
- `generate_id_dataset.sh` - Single-person parallel generation
- `generate_id_dataset_multi.sh` - Multi-person generation + auto-merge
- `merge_datasets.py` - Manual merge utility (for custom workflows)
- `monitor_progress.sh` - Progress monitoring
- `identity_stage1.py` + `processor_stage1.py` - Stage 1 (supports `--multi_person` flag)
- `identity_stage2.py` + `processor_stage2.py` - Stage 2 (handles both single & multi Q&A)

**Samples:**
- `identity_qa_pairs_sample.json` - Single-person examples (6 samples, 1 tool call each)
- `identity_qa_pairs_multi_test.json` - Multi-person examples (3 samples, 2-3 tool calls each)
- `stage1_test/` → `identity_qa_pairs_test.json` - Example outputs
- `QUALITY_COMPARISON.md` - Quality validation

**Reference:** `../refcoco/refcoco_main3.py`, `../utils/processor_main.py` (for future VLM implementation)

## Production Usage

### Single-Person Dataset (31k samples)

**Parallel Generation**
```bash
# Edit GPUs in generate_id_dataset.sh if needed
./generate_id_dataset.sh

# Monitor progress in another terminal
./monitor_progress.sh
```

**Sngle GPU (Manual)**
```bash
# Stage 1: Generate raw Q&A
CUDA_VISIBLE_DEVICES=1 python identity_stage1.py \
  --merged_data merged_refcoco_data.pkl \
  --coco_path /mnt/nas3/Data/coco \
  --start 0 --end -1

# Stage 2: Refine and enrich
python identity_stage2.py \
  --stage1_folder /mnt/nas3/Data/coco/refcoco_identity_stage1 \
  --output dataset/identity_qa_pairs_31k.json
```

### Multi-Person Dataset (14k samples, 2+ tool calls per Q&A)

**Multi-person questions require identifying 2+ people sequentially**

**Parallel Generation**
```bash
# Generates multi-person Q&A + auto-merges with single-person dataset
./generate_id_dataset_multi.sh
# Output: dataset/identity_qa_pairs_45k_complete.json (31k single + 14k multi)
```

**Sngle GPU (Manual)**
```bash
# Stage 1: Generate multi-person Q&A (--multi_person flag)
CUDA_VISIBLE_DEVICES=1 python identity_stage1.py \
  --merged_data merged_refcoco_data.pkl \
  --coco_path /mnt/nas3/Data/coco \
  --output_folder /mnt/nas3/Data/coco/refcoco_identity_stage1_multi \
  --multi_person \
  --start 0 --end -1

# Stage 2: Refine (same script, auto-detects multi-person Q&A)
python identity_stage2.py \
  --stage1_folder /mnt/nas3/Data/coco/refcoco_identity_stage1_multi \
  --output dataset/identity_qa_pairs_multi_14k.json

# Stage 3: Merge with single-person dataset
python merge_datasets.py \
  --single_person dataset/identity_qa_pairs_31k.json \
  --multi_person dataset/identity_qa_pairs_multi_14k.json \
  --output dataset/identity_qa_pairs_45k_complete.json \
  --analyze
```

**Multi-Person Question Types:**
- **Group**: "Who are all the people visible in this image?" (identifies all N people)
- **Selective**: "Who are the two people sitting on the bench?" (identifies subset)
- **Comparative**: "Between the two people, who is taller?" (compares two people)
- **Sequential**: "Who are the people from left to right?" (ordered identification)

**Configuration:**
- Single-person: ~31k samples (1 tool call/Q&A, 14k images)
- Multi-person: ~14k samples (2.3 tool calls/Q&A, 10.8k images with 2+ people)
- **Merged: ~45k samples (1.4 tool calls/Q&A, 14k images)**
