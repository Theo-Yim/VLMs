#!/bin/bash

# InternVL 3.5 LoRA Fine-tuning Script  
# Usage: bash run_training.sh

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your available GPUs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

# Training parameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
# CONFIG_FILE="${SCRIPT_DIR}/train_config.json"
CONFIG_FILE=${1:-"./InternVL3/src_theo/lora/train_config_lora.json"}
# TRAIN_SCRIPT="${SCRIPT_DIR}/train_theo_lora.py"
TRAIN_SCRIPT="./InternVL3/src_theo/lora/train_theo_lora.py"

echo "🚀 Starting InternVL 3.5 LoRA Fine-tuning..."
echo "📁 Script directory: ${SCRIPT_DIR}"
echo "⚙️  Configuration file: ${CONFIG_FILE}"
echo "🐍 Training script: ${TRAIN_SCRIPT}"

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

echo "✅ All files found, starting training..."

# Determine number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Run training - following your existing Ovis pattern
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "🏃 Running single-GPU training..."
    python "$TRAIN_SCRIPT" "$CONFIG_FILE"
else
    echo "🏃 Using torchrun for multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS "$TRAIN_SCRIPT" "$CONFIG_FILE"
fi

echo "🎉 Training completed successfully!"
echo "📁 Model saved to: $(python -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])")"
