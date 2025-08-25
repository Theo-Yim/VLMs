#!/bin/bash

# Ovis2.5 Fine-tuning Launch Script
# Usage: ./run_training.sh [config_file] [additional_args...]

set -e

# Default config file
CONFIG_FILE=${1:-"./Ovis/src_theo/train_config.json"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or specify an existing one."
    echo "Usage: $0 [config_file] [additional_args...]"
    exit 1
fi

# # Shift to get additional arguments
# shift || true

# echo "=== Ovis2.5 Fine-tuning ==="
# echo "Config file: $CONFIG_FILE"
# echo "Additional args: $@"
# echo "=========================="

# # Check GPU availability
# if command -v nvidia-smi &> /dev/null; then
#     echo "GPU Status:"
#     nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
#     echo "=========================="
# fi

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Run training
python ./Ovis/src_theo/train_theo.py "$CONFIG_FILE" # "$@"

echo "Training script completed!"