#!/bin/bash

# Ovis2.5 LoRA Fine-tuning Launch Script
# Usage: ./train_launch_lora.sh [config_file]

set -e

# Default config file
CONFIG_FILE=${1:-"./Ovis/src_theo/lora/train_config_lora.json"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or specify an existing one."
    echo "Usage: $0 [config_file]"
    exit 1
fi

echo "=== Ovis2.5 LoRA Fine-tuning ==="
echo "Config file: $CONFIG_FILE"
echo "=========================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo "=========================="
fi

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Install required packages if not already installed
echo "Checking required packages..."
python -c "import trl, peft" 2>/dev/null || {
    echo "Installing TRL and PEFT..."
    pip install trl peft
}

# Run LoRA training
echo "Starting LoRA training..."
python ./Ovis/src_theo/lora/train_theo_lora.py "$CONFIG_FILE"

echo "LoRA training completed!"
echo ""
echo "To merge LoRA adapters with base model, run:"
echo "python ./Ovis/src_theo/lora/merge_lora_adapters.py --adapter_path [checkpoint_path] --output_path [merged_model_path]"