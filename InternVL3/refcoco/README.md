# RefCOCO Synthetic VLM Dataset Generation

This project creates high-quality synthetic Vision-Language Model (VLM) training datasets by processing and enhancing RefCOCO datasets with multimodal reasoning capabilities using InternVL3.

## Overview

The pipeline intelligently merges referring expressions from RefCOCO, RefCOCOplus, and RefCOCOg datasets, then generates sophisticated question-answer pairs with multimodal interleaved reasoning. The generated dataset supports training VLMs with enhanced visual understanding and reasoning capabilities.

### Summary

1. **merge_refcoco_datasets.py**: Merges RefCOCO datasets with semantic similarity filtering → `merged_refcoco_data.pkl`
2. **refcoco_main3.py**: Generates Q, A1, A2 → Individual JSON files
3. **refcoco_main3_verify.py**: Fixes missing A2 responses (optional)
4. **refcoco_main_llm.py**: Generates refined A3 responses, with automated verification and fixing of incomplete tool uses and response format.
5. **create_jsonl.py**: Converts to training JSONL format with Q and A3
6. **convert_QnA_data_to_standard.py**: Converts to standard conversation format for immediate training use

## Usage

### Step 1: Dataset Merging

Create unified dataset by merging referring expressions from multiple RefCOCO datasets:

```bash
python merge_refcoco_datasets.py
```

**Output**: `merged_refcoco_data.pkl`

**Features**:
- Merges RefCOCO, RefCOCOplus, and RefCOCOg datasets
- Intelligent referring expression ranking and deduplication
- Semantic similarity-based filtering
- Enhanced descriptions with spatial and contextual information

### Step 2: Question-Answer Generation

Generate initial questions and detailed reasoning responses:

```bash
python refcoco_main3.py
```

**TODO**: Check and adjust the location output_path="/mnt/nas3/Data/coco" when calling RefCOCOProcessor

**Output**: Individual JSON files in `/mnt/nas3/Data/coco/refcoco_vlm_results_theo/`

**Generated Content**:
- **Q**: Questions focused on specific objects
- **A1**: Initial direct answers
- **A2**: Detailed multimodal reasoning responses with crop tool usage

### Step 3: Quality Verification

Verify and fix missing or incomplete A2 responses:

```bash
python refcoco_main3_verify.py
```

**Features**:
- Automated detection of missing/incomplete responses
- Batch processing with progress tracking
- Comprehensive error reporting

### Step 4: LLM Enhancement

Generate refined A3 responses using LLM:

```bash
python refcoco_main_llm.py
```

**TODO**: Check and adjust the location output_path="/mnt/nas3/Data/coco" when calling RefCOCOProcessor

**Output**: Enhanced JSON files in `/mnt/nas3/Data/coco/refcoco_vlm_results_theo_llm/`

**Features**:
- LLM-based answer refinement
- Format standardization
- Response quality improvement

### Step 5: Training Format Generation

Convert to training-ready JSONL format:

```bash
python create_jsonl.py --input_dir /path/to/json/files --output_file refcoco_qa_pairs.jsonl
```

**Options**:
- `--preview`: Show sample entries without generating full output
- `--input_dir`: Directory containing JSON files (default: `/mnt/nas3/Data/coco/refcoco_vlm_results_theo_llm/`)
- `--output_file`: Output JSONL file path (default: `refcoco_qa_pairs.jsonl`)

### Step 6: Conversation Format Conversion

Convert QnA format to standard conversation format for immediate training use:

```bash
python convert_QnA_data_to_standard.py refcoco_qa_pairs.jsonl refcoco_train_ready.json
```

**Features**:
- Converts QnA format to standard "human"/"gpt" conversation format
- Processes tool calls: `{Crop ...}` → `<tool_call>Crop ...</tool_call>`
- Ready for direct use in VLM training pipelines

**Options**:
- `--image_base_path`: Base path for resolving relative image paths

## File Structure

```
refcoco/
├── README.md                     # This file
├── merge_refcoco_datasets.py     # Dataset merging and preprocessing
├── refcoco_main3.py             # Main QA generation pipeline
├── refcoco_main3_verify.py      # Response verification and fixing
├── refcoco_main_llm.py          # LLM-based answer enhancement
├── create_jsonl.py              # Training format conversion
├── convert_QnA_data_to_standard.py # Conversation format conversion
└── extras/                      # Experimental code (not in active use)
    ├── inference_*.py           # Various inference experiments
    ├── refcoco_main*.py         # Alternative pipeline versions
    ├── questions.py             # Sample question templates
    └── ref_eval_sample_kr.py    # Evaluation utilities
```

## Output Format

### Individual JSON Files
```json
{
  "image_path": "coco/train2017/000000581738.jpg",
  "image_id": "000000581738",
  "annos_str": "- bus 1 [15.21, 165.15, 189.90, 291.08], which is \"the red bus whose front end is blocked from view\", \"the orange bus with a yellow rear view mirror\", \"red bus you dont see the front of\", \"red bus with windshield hidden\", \"the red bus thats blocked in\"",
  "QnA": [
    {
      "Q": "What notable feature can be seen on the rear of the bus that is partially visible in the image?",
      "A1": "The bus has a yellow rearview mirror.",
      "A2": "<think>\nTo identify the notable feature on the rear of the partially visible bus, I need to examine that part of the image closely. Let me crop the region to get a better look.\n\n{Crop bus 1}\n\nUpon closer inspection of the cropped area, I can confirm that there is a yellow rearview mirror visible on the rear side of the bus.\n</think>\n\n<answer>\nThe notable feature on the rear of the partially visible bus is a yellow rearview mirror.\n</answer>",
      "A3": "<think>\nTo identify the notable feature on the rear of the partially visible bus, I need to examine that part of the image closely. Let me crop the region to get a better look.\n\n{Crop bus 1 [15.21, 165.15, 189.90, 291.08]}\n\nUpon closer inspection of the cropped area, I can confirm that there is a yellow rearview mirror visible on the rear side of the bus.\n</think>\n\n<answer>\nThe notable feature on the rear of the partially visible bus is a yellow rearview mirror.\n</answer>"
    }
  ]
}
```

### Training JSONL Format
```json
{"image_path": "coco/train2017/000000549347.jpg", "image_id": "000000549347", "QnA": [{"Q": "What is the individual doing?", "A3": "<think>Let me closely look at the person.\n\n{Crop person 2 [181, 16, 220, 191]}\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"}]}
```

### Final Conversation Format (Training-Ready)
```json
{
  "image": "coco/train2017/000000549347.jpg",
  "image_id": "000000549347", 
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is the individual doing?"
    },
    {
      "from": "gpt",
      "value": "<think>Let me closely look at the person.\n\n<tool_call>Crop person 2 [181, 16, 220, 191]</tool_call>\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.</answer>"
    }
  ]
}
```

### Features
Implements sophisticated reasoning patterns with crop tool use:
```
<think>
To answer this question, I need to examine the visual cues...
{Crop person 1}
From the cropped view, I can see...
</think>
<answer>
Based on the visual analysis...
</answer>
```

## Notes

- Models: InternVL3 (main), Qwen (LLM enhancement)
- The `extras/` directory contains experimental code for reference