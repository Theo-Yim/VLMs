#!/bin/bash

# Ovis2.5 Fine-tuning Launch Script with DeepSpeed Support
# Usage: ./train_launch.sh [config_file] [num_gpus] [script_type]
# 
# Arguments:
#   config_file: Path to training config JSON (default: ./Ovis/src_theo/train_config.json)
#   num_gpus: Number of GPUs to use (default: auto-detect)
#   script_type: 'standard' or 'trl' (default: standard)

export PYTHONPATH=/workspace/VLMs/Ovis:$PYTHONPATH

set -e

# Default values
CONFIG_FILE=${1:-"./Ovis/src_theo/train_config.json"}
NUM_GPUS=${2:-0}  # 0 means auto-detect
SCRIPT_TYPE=${3:-"standard"}  # standard or trl

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Please create a configuration file or specify an existing one."
    echo "Usage: $0 [config_file] [num_gpus] [script_type]"
    exit 1
fi

echo "=== Ovis2.5 Fine-tuning with DeepSpeed ==="
echo "Config file: $CONFIG_FILE"
echo "Script type: $SCRIPT_TYPE"
echo "========================================="

# Auto-detect GPU count if not specified
if [ "$NUM_GPUS" -eq 0 ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPU(s)"
    else
        NUM_GPUS=1
        echo "nvidia-smi not found, defaulting to 1 GPU"
    fi
else
    echo "Using $NUM_GPUS GPU(s) as specified"
fi

# Check GPU availability and status
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    echo "========================================="
    
    # Verify requested GPUs are available
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

# Determine which training script to use
if [ "$SCRIPT_TYPE" = "trl" ]; then
    TRAIN_SCRIPT="./Ovis/src_theo/train_theo_trl.py"
    echo "Using TRL SFTTrainer script"
else
    TRAIN_SCRIPT="./Ovis/src_theo/train_theo.py"
    echo "Using standard Trainer script"
fi

# Check if training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found!"
    exit 1
fi

echo "Training script: $TRAIN_SCRIPT"
echo "========================================="

# Install required packages if not already installed
echo "Checking required packages..."
if [ "$SCRIPT_TYPE" = "trl" ]; then
    python -c "import trl, peft" 2>/dev/null || {
        echo "Installing TRL and PEFT..."
        pip install trl peft
    }
fi

# Run training with appropriate launcher
echo "Starting training with $NUM_GPUS GPU(s)..."
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU - direct Python call
    echo "Single GPU training mode"
    python "$TRAIN_SCRIPT" "$CONFIG_FILE"
else
    # Multi-GPU - use torchrun with DeepSpeed
    echo "Multi-GPU training mode with torchrun"
    torchrun --nproc_per_node=$NUM_GPUS "$TRAIN_SCRIPT" "$CONFIG_FILE"
fi

echo "========================================="
echo "Training completed successfully!"
echo ""
echo "Output directory configured in: $CONFIG_FILE"
if [ "$SCRIPT_TYPE" = "trl" ] && grep -q "peft_config" "$TRAIN_SCRIPT"; then
    echo "For LoRA training, use the merge script to combine adapters with base model."
fi