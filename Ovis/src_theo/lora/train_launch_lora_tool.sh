#!/bin/bash

# Ovis2.5 LoRA Fine-tuning Launch Script
# Usage: ./train_launch_lora.sh [config_file] [num_gpus]

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on your available GPUs
export PYTHONPATH=/workspace/VLMs/Ovis:$PYTHONPATH

set -e

# Default config file
CONFIG_FILE=${1:-"./Ovis/src_theo/lora/train_config_lora_tool.json"}

# Determine number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or specify an existing one."
    echo "Usage: $0 [config_file] [num_gpus]"
    exit 1
fi

echo "=== Ovis2.5 LoRA Fine-tuning ==="
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo "=========================="
    
    # Check if requested number of GPUs is available
    available_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NUM_GPUS" -gt "$available_gpus" ]; then
        echo "Warning: Requested $NUM_GPUS GPUs but only $available_gpus are available."
        echo "Using $available_gpus GPUs instead."
        NUM_GPUS=$available_gpus
    fi
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

# Run LoRA training with torchrun for multi-GPU support
echo "Starting LoRA training with $NUM_GPUS GPU(s)..."
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU - use direct Python call
    python ./Ovis/src_theo/lora/train_theo_lora.py "$CONFIG_FILE"
else
    # Multi-GPU - use torchrun
    torchrun --nproc_per_node=$NUM_GPUS ./Ovis/src_theo/lora/train_theo_lora.py "$CONFIG_FILE"
fi

echo "LoRA training completed!"
echo ""
echo "To merge LoRA adapters with base model, run:"
echo "python ./Ovis/src_theo/lora/merge_lora_adapters.py --adapter_path [checkpoint_path] --output_path [merged_model_path]"