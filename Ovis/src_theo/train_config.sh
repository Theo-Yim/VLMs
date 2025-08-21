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
LEARNING_RATE=2e-5
MULTIMODAL_MAX_LENGTH=8192
TEXT_MAX_LENGTH=4096

# Image processing parameters (original Ovis defaults)
SINGLE_IMAGE_MIN_PIXELS=$((448*448))
SINGLE_IMAGE_MAX_PIXELS=$((1792*1792))
MULTIPLE_IMAGE_MIN_PIXELS=$((448*448))
MULTIPLE_IMAGE_MAX_PIXELS=$((896*896))
VIDEO_MIN_PIXELS=$((448*448))
VIDEO_MAX_PIXELS=$((896*896))

# ViT configuration (original defaults)
VIT_HIDDEN_STRIDE=2
VIT_WINDOW_SIZE=112
VIT_TEMPORAL_PATCH_SIZE=1
VISUAL_VOCAB_SIZE=65536
