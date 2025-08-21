# Ovis Multi-Stage Training Guide

This guide explains how to use the `stage` parameter for progressive training of Ovis models, following the original Ovis training methodology.

## Understanding Training Stages

### Stage 1: Visual Tokenizer Training
**Purpose**: Train the visual encoder to understand images  
**Parameters**: `--stage 1 --train_modules "visual_tokenizer"`
- Only visual tokenizer (ViT) parameters are trained
- LLM parameters are frozen
- Focus on visual feature extraction
- Typical learning rate: 1e-4
- Duration: 1-2 epochs

```bash
python train_theo.py \
    --stage 1 \
    --train_modules "visual_tokenizer" \
    --learning_rate 1e-4 \
    --num_train_epochs 2
```

### Stage 2: Visual-Text Alignment Training  
**Purpose**: Train the connection between visual and text embeddings  
**Parameters**: `--stage 2 --train_modules "visual_tokenizer|vte"`
- Visual tokenizer + visual text embedding (VTE) are trained
- LLM parameters remain frozen
- Focus on multimodal alignment
- Typical learning rate: 5e-5
- Duration: 1-2 epochs

```bash
python train_theo.py \
    --stage 2 \
    --train_modules "visual_tokenizer|vte" \
    --ovis_pretrained_path "./checkpoints/stage1_model" \
    --learning_rate 5e-5 \
    --num_train_epochs 2
```

### Stage 3: Full Model Fine-tuning
**Purpose**: End-to-end fine-tuning of the complete model  
**Parameters**: `--stage 3 --train_modules "all"`
- All parameters (LLM + visual components) are trained
- Complete multimodal understanding
- Typical learning rate: 2e-5
- Duration: 2-3 epochs

```bash
python train_theo.py \
    --stage 3 \
    --train_modules "all" \
    --ovis_pretrained_path "./checkpoints/stage2_model" \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

## Advanced Module Training

### Custom Learning Rates per Module
You can specify different learning rates for different modules:

```bash
# Visual tokenizer head with higher LR, VTE with lower LR
--train_modules "visual_tokenizer.head:1e-3|vte:5e-5"

# Different rates for visual and language components
--train_modules "visual_tokenizer:1e-4|llm:2e-5"
```

### Available Training Modules

| Module | Description | Typical Use Case |
|--------|-------------|------------------|
| `all` | All model parameters | Stage 3 full fine-tuning |
| `llm` | Language model only | Text adaptation |
| `visual_tokenizer` | Complete visual encoder | Stage 1 visual training |
| `visual_tokenizer.head` | Visual tokenizer head only | Quick visual adaptation |
| `visual_tokenizer.vit` | Vision transformer only | Pure visual feature training |
| `vte` | Visual text embedding | Stage 2 alignment training |

### Combining Modules
Use `|` to combine multiple modules:
```bash
--train_modules "visual_tokenizer|vte"           # Stage 2 typical
--train_modules "visual_tokenizer.head|llm"      # Custom combination
--train_modules "visual_tokenizer:1e-4|llm:2e-5" # With different LRs
```

## Progressive Training Workflow

### Full 3-Stage Training
```bash
# Stage 1: Visual understanding
./stage_configs.sh stage1

# Stage 2: Visual-text alignment  
./stage_configs.sh stage2

# Stage 3: Full fine-tuning
./stage_configs.sh stage3

# Or run all at once
./stage_configs.sh all
```

### Quick Training (Single Stage)
For smaller datasets or quick experiments:
```bash
# Direct stage 3 training (most common)
python train_theo.py \
    --stage 3 \
    --train_modules "all" \
    --llm_name_or_path "microsoft/DialoGPT-medium" \
    --num_train_epochs 3
```

### Adapter-Style Training
For parameter-efficient training:
```bash
# Train only visual components, freeze LLM
python train_theo.py \
    --stage 2 \
    --train_modules "visual_tokenizer|vte" \
    --freeze_llm True \
    --learning_rate 1e-4
```

## Stage-Specific Recommendations

### Stage 1 Settings
- **Learning Rate**: 1e-4 (higher for visual components)
- **Batch Size**: Can be larger (8-16) since only visual training
- **Epochs**: 1-2 epochs sufficient
- **Data**: Can use image-caption pairs efficiently

### Stage 2 Settings  
- **Learning Rate**: 5e-5 (moderate for alignment)
- **Batch Size**: 4-8 (multimodal processing)
- **Epochs**: 1-2 epochs
- **Data**: Requires image-text conversation pairs

### Stage 3 Settings
- **Learning Rate**: 2e-5 (lower for stability)
- **Batch Size**: 2-4 (memory intensive)
- **Epochs**: 2-3 epochs
- **Data**: Full conversation datasets

## Monitoring Training

### Key Metrics to Watch
- **Stage 1**: Visual loss convergence
- **Stage 2**: Alignment quality, perplexity
- **Stage 3**: Overall conversation quality, BLEU scores

### TensorBoard Monitoring
```bash
tensorboard --logdir ./checkpoints/stage3_model/logs
```

### Common Issues and Solutions

1. **Memory Issues**
   ```bash
   # Reduce batch size, increase gradient accumulation
   --per_device_train_batch_size 2 \
   --gradient_accumulation_steps 16
   ```

2. **Training Instability**
   ```bash
   # Lower learning rate, add warmup
   --learning_rate 1e-5 \
   --warmup_ratio 0.1
   ```

3. **Slow Convergence**
   ```bash
   # Ensure proper stage progression
   # Check if previous stage model is loaded correctly
   --ovis_pretrained_path "./checkpoints/previous_stage"
   ```

## Best Practices

1. **Always validate data** before training each stage
2. **Save checkpoints frequently** during each stage
3. **Monitor memory usage** and adjust batch sizes
4. **Use appropriate learning rates** for each stage
5. **Progressive training** generally works better than single-stage
6. **Validate model quality** after each stage before proceeding

## Example Complete Workflow

```bash
# 1. Prepare data
python data_utils.py validate ./data/train_data.json ./data/images

# 2. Stage 1: Visual tokenizer
python train_theo.py --stage 1 --train_modules "visual_tokenizer" \
    --output_dir "./checkpoints/stage1" --learning_rate 1e-4 --num_train_epochs 2

# 3. Stage 2: Alignment
python train_theo.py --stage 2 --train_modules "visual_tokenizer|vte" \
    --ovis_pretrained_path "./checkpoints/stage1" \
    --output_dir "./checkpoints/stage2" --learning_rate 5e-5 --num_train_epochs 2

# 4. Stage 3: Full fine-tuning
python train_theo.py --stage 3 --train_modules "all" \
    --ovis_pretrained_path "./checkpoints/stage2" \
    --output_dir "./checkpoints/final" --learning_rate 2e-5 --num_train_epochs 3
```

This progressive approach typically yields better results than single-stage training, especially for complex multimodal tasks.