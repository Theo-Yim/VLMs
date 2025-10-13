# Crop Tool Dataset Filtering Report

## Overview

This document summarizes the dataset quality improvements achieved through systematic filtering of the RefCOCO crop tool calling dataset.

## Original Dataset Issues

### Initial Quality Assessment
- **File**: `/mnt/nas3/Data/coco/refcoco_vlm_results_theo_ready_to_train/refcoco_qa_pairs_croptool.json`
- **Total Samples**: 140,841
- **Quality Score**: **83.2/100**

### Problems Identified

1. **Missing Crop Calls** (16.1%)
   - 22,694 samples had no crop tool calls despite having system prompt instructing tool use
   - Example: Questions about wine bottle positions answered without visual inspection
   - Impact: Model learns to ignore available tools

2. **Inconsistent Crop Mentions** (13.5%)
   - 18,979 samples mentioned using crop tool in `<think>` tags but never actually called it
   - Example: "I will crop the area..." but no `<tool_call>Crop [...]</tool_call>`
   - Impact: Model learns inconsistent behavior between reasoning and action

3. **Easy Questions** (0.04%)
   - 53 questions answerable without close inspection
   - Examples: "How many people are in the image?", "What color is the background?"
   - Impact: Teaches model to use crop tool unnecessarily

4. **Verbosity** (92.9%)
   - 130,749 answers exceed 500 characters
   - Note: This is acceptable for reasoning tasks, not filtered

## Filtering Strategy

### Conservative Approach

We implemented **conservative filtering** to preserve samples that could benefit from cropping:

1. **Strict Crop Requirement**
   - Must contain at least one `<tool_call>Crop [x,y,x2,y2]</tool_call>`
   - No exceptions

2. **Consistency Check**
   - Remove samples mentioning crop in reasoning without actual tool call
   - Ensures alignment between reasoning and actions

3. **Minimal Easy Question Filtering**
   - Only removes EXTREMELY obvious cases (e.g., "How many people are visible?")
   - When in doubt, keep the sample
   - Preserves nuanced questions that might benefit from closer look

### Implementation

**Script**: `filter_dataset.py`

```python
# Filtering criteria (all must pass):
1. has_crop_call(answer) == True
2. has_crop_mentions_without_call(answer) == False
3. is_very_easy_question(question) == False (optional)
```

## Results

### Filtered Dataset
- **File**: `/mnt/nas3/Data/coco/refcoco_vlm_results_theo_ready_to_train/refcoco_qa_pairs_croptool_filtered.json`
- **Total Samples**: 117,867 (83.7% retained)
- **Quality Score**: **90.6/100** ✅ (+7.4 points)

### Filtering Breakdown

| Issue | Removed | Percentage |
|-------|---------|------------|
| No crop call | 22,694 | 16.1% |
| Mentions without call | 235 | 0.2% |
| Very easy questions | 45 | 0.03% |
| **Total filtered** | **22,974** | **16.3%** |
| **Kept** | **117,867** | **83.7%** |

### Quality Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Crop tool usage** | 83.9% | **100%** | +16.1% |
| **Consistent reasoning** | 86.5% | **100%** | +13.5% |
| **Easy questions** | 0.04% | **0%** | -0.04% |
| **Overall quality** | 83.2 | **90.6** | **+7.4 pts** |

## Sample Quality Examples

### Sample 1: Giraffe Size Comparison
```
Q: What is notable about the size of the giraffe standing in the background compared to others?

A: <think>
To determine what is notable about the size of the giraffe standing in the
background compared to others, I should first identify and compare the sizes.
<tool_call>Crop [115.97, 94.32, 377.95, 290.74]</tool_call><image>
Upon examining the image, I notice that the smallest giraffe that is standing
appears significantly smaller than the others...
</think>

<answer>
The giraffe in the background is notably smaller, indicating it is likely a juvenile.
</answer>
```

### Sample 2: Donut Type Identification
```
Q: What type of donuts are in the top bowl?

A: <think>
To determine the type of donuts in the top bowl, I need to examine the image closely.
<tool_call>Crop [133.72, 187.54, 351.92, 242.53]</tool_call><image>
Upon closer inspection, the donuts in the top bowl are covered with colorful sprinkles...
</think>

<answer>
The donuts in the top bowl are frosted donuts with sprinkles.
</answer>
```

## Remaining Characteristics

### Verbosity (Not Filtered)
- **Percentage**: 93.7% of answers exceed 500 characters
- **Reason**: Kept intentionally
- **Rationale**:
  - Crop tool reasoning requires detailed explanations
  - Shows model thinking process
  - Demonstrates why tool use was necessary
  - Valuable for training interpretable models

### Multi-Crop Usage
- **Samples with multiple crops**: 9,207 (7.8%)
- **Distribution**:
  - 1 crop: 92.2%
  - 2 crops: 6.1%
  - 3 crops: 1.3%
  - 4+ crops: 0.4%

## Training Recommendations

### What This Dataset Teaches

1. **Consistent Tool Use**
   - Every sample demonstrates proper tool calling
   - Model learns when and how to use crop tool
   - No ambiguity between reasoning and action

2. **Visual Grounding**
   - All answers based on visual inspection
   - Model learns to ground responses in image regions
   - Spatial reasoning with bbox coordinates

3. **Reasoning Transparency**
   - `<think>` tags show decision process
   - Tool calls embedded in reasoning flow
   - Model learns to explain its actions

### Expected Model Capabilities

After training on this filtered dataset, the model should:

1. ✅ **Consistently use crop tool when system prompt indicates availability**
2. ✅ **Align reasoning with actions** (no "I will crop..." without actual crop)
3. ✅ **Make appropriate tool-use decisions** (crop when closer inspection needed)
4. ✅ **Generate well-structured responses** with `<think>` and `<answer>` tags

## Comparison to Literature

### Novelty Aspects

1. **Vision-Native Tool Calls**
   - Crop tool is unique to VLMs (LLMs can't "look closer")
   - 100% tool-use consistency (higher than most agent datasets)

2. **Dataset Scale**
   - 117,867 high-quality samples
   - Covers diverse visual scenarios (RefCOCO/+/g datasets)

3. **Quality Filtering**
   - Systematic filtering improves consistency
   - Conservative approach preserves nuanced cases
   - Automated validation ensures reproducibility

## Files Generated

### Scripts
1. `validate_dataset.py` - Quality assessment tool
2. `filter_dataset.py` - Filtering implementation

### Data
1. **Original**: `refcoco_qa_pairs_croptool.json` (140,841 samples)
2. **Filtered**: `refcoco_qa_pairs_croptool_filtered.json` (117,867 samples) ✅ **Use this for training**

## Usage

### Validate Dataset
```bash
python InternVL3/refcoco/validate_dataset.py \
    --file_path /path/to/dataset.json \
    --output_issues issues.json
```

### Filter Dataset
```bash
python InternVL3/refcoco/filter_dataset.py \
    --input_file /path/to/input.json \
    --output_file /path/to/output.json \
    --keep_easy  # Optional: keep easy questions
```

## Conclusions

1. **High-Quality Dataset Ready**: 117,867 samples with 100% tool-calling consistency
2. **Quality Score: 90.6/100**: Significant improvement from 83.2
3. **Training Ready**: Filtered dataset suitable for VLM tool-calling training
4. **Reproducible Process**: Scripts provided for validation and filtering

The filtered dataset represents a high-quality resource for training VLMs with vision-native tool-calling capabilities, with systematic quality control and conservative filtering to preserve valuable training examples.
