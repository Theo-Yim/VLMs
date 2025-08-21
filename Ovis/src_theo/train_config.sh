#!/bin/bash

# Training configuration script for Ovis custom training
# Based on original Ovis training parameters - clean and simple

# Data paths
DATA_PATH="./data/train_data.json"
IMAGE_FOLDER="./data/images"

# Model paths (CORRECT for Ovis2.5-9B)
LLM_MODEL="Qwen/Qwen3-8B"  # Correct LLM for Ovis2.5-9B
VIT_MODEL="google/siglip2-so400m-patch16-512"  # Correct ViT for Ovis2.5-9B
OVIS25_MODEL="AIDC-AI/Ovis2.5-9B"

# Output directory
OUTPUT_DIR="./checkpoints/theo_ovis_training"

# Training parameters (original Ovis defaults)
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MULTIMODAL_MAX_LENGTH=4096
TEXT_MAX_LENGTH=4096

# Image processing parameters (original Ovis defaults)
SINGLE_IMAGE_MIN_PIXELS=$((448*448))
SINGLE_IMAGE_MAX_PIXELS=$((1792*1344))
MULTIPLE_IMAGE_MIN_PIXELS=$((448*448))
MULTIPLE_IMAGE_MAX_PIXELS=$((448*448))
VIDEO_MIN_PIXELS=$((448*448))
VIDEO_MAX_PIXELS=$((448*448))

# Training modules
TRAIN_MODULES="all"

# ViT configuration (original defaults)
VIT_HIDDEN_STRIDE=2
VIT_WINDOW_SIZE=112
VIT_TEMPORAL_PATCH_SIZE=1
VISUAL_VOCAB_SIZE=65536

# Other parameters
MONITOR_STEP=100
SAVE_STEPS=500
LOGGING_STEPS=10
STAGE=3

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting Ovis training with the following configuration:"
echo "Data path: $DATA_PATH"
echo "Image folder: $IMAGE_FOLDER"
echo "LLM model: $LLM_MODEL (Ovis2.5-9B compatible)"
echo "ViT model: $VIT_MODEL (Ovis2.5-9B compatible)"
echo "Output directory: $OUTPUT_DIR"
echo "Training modules: $TRAIN_MODULES"
echo ""

# Training command (original Ovis parameters)
python train_theo.py \
    --llm_name_or_path $LLM_MODEL \
    --vit_name_or_path $VIT_MODEL \
    --visual_vocab_size $VISUAL_VOCAB_SIZE \
    --conversation_formatter_class "Qwen3ConversationFormatter" \
    --vit_hidden_stride $VIT_HIDDEN_STRIDE \
    --vit_window_size $VIT_WINDOW_SIZE \
    --vit_temporal_patch_size $VIT_TEMPORAL_PATCH_SIZE \
    --vit_preserve_original_pe True \
    --vit_use_rope True \
    --accepts_loss_kwargs True \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --data_type "conversation" \
    --stage $STAGE \
    --multimodal_max_length $MULTIMODAL_MAX_LENGTH \
    --text_max_length $TEXT_MAX_LENGTH \
    --single_image_min_pixels $SINGLE_IMAGE_MIN_PIXELS \
    --single_image_max_pixels $SINGLE_IMAGE_MAX_PIXELS \
    --multiple_image_min_pixels $MULTIPLE_IMAGE_MIN_PIXELS \
    --multiple_image_max_pixels $MULTIPLE_IMAGE_MAX_PIXELS \
    --video_min_pixels $VIDEO_MIN_PIXELS \
    --video_max_pixels $VIDEO_MAX_PIXELS \
    --train_modules $TRAIN_MODULES \
    --monitor_step $MONITOR_STEP \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 True \
    --remove_unused_columns False \
    --report_to "tensorboard" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --optim "adamw_torch" \
    --save_safetensors True \
    --model_init_seed 0 \
    --seed 1337

echo ""
echo "Training completed! Check results in $OUTPUT_DIR"
echo "Monitor progress with: tensorboard --logdir $OUTPUT_DIR/logs"