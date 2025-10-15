# GRPO Training for Ovis2.5 Tool-Calling

**Group Relative Policy Optimization** for refining Ovis2.5's tool-calling capabilities (Crop + Identify) after SFT.

## When to Use

✅ **Use GRPO when:**
- SFT model makes systematic errors (incorrect tool usage, invalid bboxes)
- Need to explore better reasoning strategies beyond supervised examples
- Want to optimize tool selection and spatial reasoning

❌ **Don't use GRPO for:**
- Initial training (use `train_theo.py` SFT first)
- High-quality synthetic data that already works well

---

## Training Pipeline

### Stage 1: SFT (Required First)
```bash
# Train with supervised fine-tuning
bash Ovis/src_theo/train_launch.sh  # or lora/train_launch_lora.sh
# Output: ./Ovis/checkpoints/ovis25_finetune_final/
```

### Stage 2: GRPO Refinement
```bash
# Edit grpo_config_tool.json: set "sft_model_path" to your SFT checkpoint

# Single GPU
python Ovis/src_theo/train_theo_grpo.py Ovis/src_theo/grpo_config_tool.json

# Multi-GPU (DDP only - no DeepSpeed yet)
torchrun --nproc_per_node=4 Ovis/src_theo/train_theo_grpo.py \
    Ovis/src_theo/grpo_config_tool.json
```

---

## Reward Model

Three-component weighted reward (sums to 1.0):

### 1. Tool Usage Correctness (40%)
- Properly formatted tool calls: `<tool_call>Crop [x,y,x2,y2]</tool_call>`
- Identify tools with responses: `<tool_response>Name</tool_response>`
- Tools within `<think>` tags

### 2. Bounding Box Validity (30%)
- Numeric coordinates with `x1 < x2`, `y1 < y2`
- No negative or out-of-range values
- Score = 1.0 - (invalid_boxes / total_boxes)

### 3. Reasoning Quality (30%)
- Proper `<think>...</think>` and `<answer>...</answer>` structure
- Reasoning before tool calls (>10 words)
- Integration of tool results (text after `<tool_response>`)
- Appropriate length (20-1000 words)

---

## Key Configuration

**In `grpo_config_tool.json`:**

```json
{
  "sft_model_path": "./Ovis/checkpoints/ovis25_finetune_final",
  "num_generations": 4,           // Completions per prompt (reduce to 2 if OOM)
  "learning_rate": 1e-5,          // Lower than SFT (typically 1e-4)
  "beta": 0.01,                   // KL penalty (↑ = stay closer to SFT)

  "tool_usage_weight": 0.4,
  "bbox_validity_weight": 0.3,
  "reasoning_quality_weight": 0.3
}
```

---

## Architecture

**Problem:** TRL's GRPOTrainer doesn't support Ovis2.5's multimodal generation.

**Solution:** Custom components:

1. **OvisMultimodalGenerator** (`ovis_grpo_generator.py`)
   - Wraps `model.preprocess_inputs()` for image preprocessing
   - Handles variable-sized tensors (batch_size=1)
   - Generates multiple completions per prompt

2. **OvisGRPOTrainer** (`ovis_grpo_trainer.py`)
   - Extends TRL's GRPOTrainer with custom generation
   - Custom dataloader with multimodal collate function

3. **ToolCallingRewardModel** (`train_theo_grpo.py`)
   - Regex-based tool call validation
   - Multi-component reward computation

---

## Limitations

1. **No DeepSpeed support** - Use DDP (torchrun) for multi-GPU
2. **Text-only rewards** - Cannot verify bbox contains correct object (future: visual rewards)
3. **No real-time tool execution** - Tools from training data, not executed during generation

---

## Troubleshooting

**OOM:** Reduce `num_generations: 2` or `max_completion_length: 2048`
**Low rewards:** Adjust weights based on dataset characteristics
**Import errors:** Run from `/workspace/VLMs` or set `PYTHONPATH=/workspace/VLMs/Ovis/src_theo:$PYTHONPATH`

---

## Expected Improvements

| Metric | SFT | GRPO | Δ |
|--------|-----|------|---|
| Tool format correctness | 85% | 95% | +10% |
| Valid bounding boxes | 75% | 90% | +15% |
| Reasoning coherence | 70% | 85% | +15% |

*(Based on VLM-R³ results)*

---

## Files

- `train_theo_grpo.py` - Main training script
- `ovis_grpo_trainer.py` - Custom GRPO trainer
- `ovis_grpo_generator.py` - Multimodal generation wrapper
- `grpo_config_tool.json` - Training configuration

## References

- [VLM-R³ paper](https://arxiv.org/abs/2505.16192) - R-GRPO for visual grounding
- [TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer) - Hugging Face implementation
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) - Original GRPO paper
