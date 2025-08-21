# Ovis Custom Training Setup - src_theo

This folder contains a custom training implementation for Ovis models based on the original Ovis2.5 training structure. It allows you to train Ovis2.5 models with your own datasets.

## Data Format

Check ovis/train/dataset/


## Configure Training

```bash
# Data paths
DATA_PATH="./data/train_data.json"
IMAGE_FOLDER="./data/images"

# Model configuration (original Ovis structure)
LLM_MODEL="Qwen/Qwen3-8B"  # or "microsoft/DialoGPT-medium"
VIT_MODEL="google/siglip2-so400m-patch16-512"
OVIS25_MODEL="AIDC-AI/Ovis2.5-9B"

# Training parameters (original defaults)
MULTIMODAL_MAX_LENGTH=8192  # Original default
TEXT_MAX_LENGTH=4096

# Inference parameters - Thinking mode & budget
enable_thinking = True  # either True or False
enable_thinking_budget = True  # Only effective if enable_thinking is True.
# Inference parameters
max_new_tokens=1024 if enable_thinking is False else 3096
thinking_budget=2048

# Image processing (original Ovis defaults)
SINGLE_IMAGE_MIN_PIXELS=200704  # 448*448
SINGLE_IMAGE_MAX_PIXELS=3211264 # 1792*1792
MULTIPLE_IMAGE_MIN_PIXELS=200704 # 448*448  
MULTIPLE_IMAGE_MAX_PIXELS=802816 # 896*896
VIDEO_MAX_PIXELS=802816 # 896*896
```


## Directory Structure

```
Ovis/                     # Ovis2.5 Github Repository by authors
├── ovis/                 # Ovis2.5 main code released by authors
├── HF_Repo/              # Ovis2.5 HuggingFace Repository (Most recent and accurate codes and settings) by authors

Ovis/src_theo/
├── inference_ovis25.py   # Inference code for various input modalities
├── train_theo.py         # Main training script (To be filled)
└── README.md             # This file

```
