#!/bin/bash

# Multi-Stage Training Configurations for Ovis
# Based on original Ovis training methodology

# Common paths
DATA_DIR="./data"
TRAIN_DATA_PATH="./data/train_data.json"
EVAL_DATA_PATH="./data/eval_data.json"
IMAGE_FOLDER="./data/images"
LLM_MODEL="microsoft/DialoGPT-medium"
VISUAL_MODEL="google/siglip-so400m-patch14-384"

# Common training parameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
MAX_LENGTH=2048
MULTIMODAL_MAX_LENGTH=8192

# ==============================================
# STAGE 1: Visual Tokenizer Training
# ==============================================
stage1_training() {
    echo "=== STAGE 1: Visual Tokenizer Training ==="
    
    OUTPUT_DIR="./checkpoints/stage1_visual_tokenizer"
    
    python train_theo.py \
        --stage 1 \
        --train_modules "visual_tokenizer" \
        --llm_name_or_path $LLM_MODEL \
        --visual_tokenizer_pretrained_path $VISUAL_MODEL \
        --multimodal_max_length $MULTIMODAL_MAX_LENGTH \
        --data_type "conversation" \
        --train_data_path $TRAIN_DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --max_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 2 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate 1e-4 \
        --weight_decay 0.01 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_total_limit 3 \
        --dataloader_num_workers 4 \
        --bf16 True \
        --remove_unused_columns False \
        --report_to "tensorboard" \
        --logging_dir "$OUTPUT_DIR/logs"
    
    echo "Stage 1 completed. Model saved in $OUTPUT_DIR"
}

# ==============================================
# STAGE 2: Visual + Connector Training
# ==============================================
stage2_training() {
    echo "=== STAGE 2: Visual + Connector Training ==="
    
    STAGE1_MODEL="./checkpoints/stage1_visual_tokenizer"
    OUTPUT_DIR="./checkpoints/stage2_visual_connector"
    
    python train_theo.py \
        --stage 2 \
        --train_modules "visual_tokenizer|vte" \
        --ovis_pretrained_path $STAGE1_MODEL \
        --multimodal_max_length $MULTIMODAL_MAX_LENGTH \
        --data_type "conversation" \
        --train_data_path $TRAIN_DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --max_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 2 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_total_limit 3 \
        --dataloader_num_workers 4 \
        --bf16 True \
        --remove_unused_columns False \
        --report_to "tensorboard" \
        --logging_dir "$OUTPUT_DIR/logs"
    
    echo "Stage 2 completed. Model saved in $OUTPUT_DIR"
}

# ==============================================
# STAGE 3: Full Model Fine-tuning
# ==============================================
stage3_training() {
    echo "=== STAGE 3: Full Model Fine-tuning ==="
    
    STAGE2_MODEL="./checkpoints/stage2_visual_connector"
    OUTPUT_DIR="./checkpoints/stage3_full_model"
    
    python train_theo.py \
        --stage 3 \
        --train_modules "all" \
        --ovis_pretrained_path $STAGE2_MODEL \
        --multimodal_max_length $MULTIMODAL_MAX_LENGTH \
        --data_type "conversation" \
        --train_data_path $TRAIN_DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --max_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 3 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 500 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_total_limit 3 \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --dataloader_num_workers 4 \
        --bf16 True \
        --remove_unused_columns False \
        --report_to "tensorboard" \
        --logging_dir "$OUTPUT_DIR/logs"
    
    echo "Stage 3 completed. Final model saved in $OUTPUT_DIR"
}

# ==============================================
# CUSTOM STAGE: Specific Module Training
# ==============================================
custom_stage_training() {
    echo "=== CUSTOM STAGE: Specific Module Training ==="
    
    # Example: Train only visual tokenizer head with different learning rates
    OUTPUT_DIR="./checkpoints/custom_head_training"
    
    python train_theo.py \
        --stage 2 \
        --train_modules "visual_tokenizer.head:1e-3|vte:5e-5" \
        --llm_name_or_path $LLM_MODEL \
        --visual_tokenizer_pretrained_path $VISUAL_MODEL \
        --multimodal_max_length $MULTIMODAL_MAX_LENGTH \
        --data_type "conversation" \
        --train_data_path $TRAIN_DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --max_length $MAX_LENGTH \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 2 \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate 5e-5 \
        --weight_decay 0.01 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --save_steps 500 \
        --dataloader_num_workers 4 \
        --bf16 True \
        --remove_unused_columns False \
        --report_to "tensorboard" \
        --logging_dir "$OUTPUT_DIR/logs"
    
    echo "Custom stage completed. Model saved in $OUTPUT_DIR"
}

# ==============================================
# Usage Functions
# ==============================================

show_usage() {
    echo "Multi-Stage Training for Ovis"
    echo "Usage: $0 [stage1|stage2|stage3|custom|all]"
    echo ""
    echo "Stages:"
    echo "  stage1  - Train visual tokenizer only"
    echo "  stage2  - Train visual tokenizer + connector"
    echo "  stage3  - Full model fine-tuning"
    echo "  custom  - Custom module training example"
    echo "  all     - Run all stages sequentially"
    echo ""
    echo "Training Modules Options:"
    echo "  all                    - Train all parameters"
    echo "  llm                    - Train LLM only"
    echo "  visual_tokenizer       - Train visual tokenizer"
    echo "  visual_tokenizer.head  - Train visual tokenizer head only"
    echo "  visual_tokenizer.vit   - Train visual tokenizer ViT only"
    echo "  vte                    - Train visual text embedding"
    echo ""
    echo "Multi-module example: --train_modules 'visual_tokenizer.head:1e-3|vte:5e-5'"
}

# ==============================================
# Main Execution
# ==============================================

case "$1" in
    "stage1")
        stage1_training
        ;;
    "stage2")
        stage2_training
        ;;
    "stage3")
        stage3_training
        ;;
    "custom")
        custom_stage_training
        ;;
    "all")
        echo "Running all stages sequentially..."
        stage1_training
        stage2_training
        stage3_training
        echo "All stages completed!"
        ;;
    *)
        show_usage
        ;;
esac