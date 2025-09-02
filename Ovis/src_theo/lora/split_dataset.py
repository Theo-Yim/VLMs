#!/usr/bin/env python3
"""
Split the training dataset into train/validation sets
"""
import json
import random
from pathlib import Path

def split_dataset(input_path, train_ratio=0.85, seed=42):
    """Split dataset into train and validation sets"""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Load original dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split point
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    
    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Save split datasets
    input_path = Path(input_path)
    train_path = input_path.parent / f"train_{input_path.name}"
    val_path = input_path.parent / f"validation_{input_path.name}"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Train dataset saved to: {train_path}")
    print(f"Validation dataset saved to: {val_path}")
    
    return train_path, val_path

if __name__ == "__main__":
    input_file = "./utils/lh-poc/training_dataset_lh_jh.json"
    split_dataset(input_file, train_ratio=0.8, seed=42)
