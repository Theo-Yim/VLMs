#!/bin/bash

# Ovis2.5 GRPO Training Launch Script for Tool-Calling
# Usage: ./train_launch_grpo.sh [config_file] [num_gpus]

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your available GPUs
export PYTHONPATH=/workspace/VLMs/Ovis:$PYTHONPATH

set -e

# Default config file
CONFIG_FILE=${1:-"./Ovis/src_theo/grpo/grpo_config_tool.json"}

# Determine number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or specify an existing one."
    echo "Usage: $0 [config_file]"
    exit 1
fi

echo "=== Ovis2.5 GRPO Training for Tool-Calling ==="
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "==========================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo "==========================================="
fi

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Install required packages if not already installed
echo "Checking required packages..."
python -c "import trl" 2>/dev/null || {
    echo "Installing TRL (with GRPO support)..."
    pip install "trl>=0.13.0"
}

python -c "import flash_attn" 2>/dev/null || {
    echo "Warning: flash-attn not installed. Install for better performance:"
    echo "pip install flash-attn --no-build-isolation"
}

# Run GRPO training
echo "Starting GRPO training with $NUM_GPUS GPU(s)..."
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    python ./Ovis/src_theo/grpo/train_theo_grpo.py "$CONFIG_FILE"
else
    # Multi-GPU - use torchrun
    torchrun --nproc_per_node=$NUM_GPUS ./Ovis/src_theo/grpo/train_theo_grpo.py "$CONFIG_FILE"
fi

echo "GRPO training completed!"
echo ""
echo "Checkpoints saved to: $(jq -r '.output_dir' $CONFIG_FILE)"
