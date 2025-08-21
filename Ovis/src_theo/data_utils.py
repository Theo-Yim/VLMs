"""
Utility functions for data preparation and validation
"""

import json
import os
from typing import Dict, List, Tuple
from PIL import Image
import argparse


def validate_data_format(data_path: str, image_folder: str) -> Tuple[bool, List[str]]:
    """
    Validate data format and check for missing images
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if data file exists
    if not os.path.exists(data_path):
        errors.append(f"Data file not found: {data_path}")
        return False, errors
    
    # Load and validate JSON
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON format: {e}")
        return False, errors
    
    # Check data structure
    if "conversations" not in data:
        errors.append("Missing 'conversations' key in data")
        return False, errors
    
    conversations = data["conversations"]
    if not isinstance(conversations, list):
        errors.append("'conversations' should be a list")
        return False, errors
    
    # Validate each conversation
    for i, conv in enumerate(conversations):
        # Check required fields
        if "id" not in conv:
            errors.append(f"Missing 'id' in conversation {i}")
        
        if "conversations" not in conv:
            errors.append(f"Missing 'conversations' in item {i}")
            continue
        
        # Check conversation format
        conv_list = conv["conversations"]
        if not isinstance(conv_list, list):
            errors.append(f"'conversations' should be a list in item {i}")
            continue
        
        for j, turn in enumerate(conv_list):
            if "from" not in turn or "value" not in turn:
                errors.append(f"Missing 'from' or 'value' in conversation {i}, turn {j}")
        
        # Check image if specified
        if "image" in conv and conv["image"]:
            image_path = conv["image"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(image_folder, image_path)
            
            if not os.path.exists(image_path):
                errors.append(f"Image not found: {image_path} (conversation {i})")
            else:
                # Try to open image
                try:
                    img = Image.open(image_path)
                    img.verify()
                except Exception as e:
                    errors.append(f"Cannot open image {image_path}: {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def convert_to_ovis_format(input_data: List[Dict], output_path: str):
    """
    Convert data to Ovis format if needed
    
    Args:
        input_data: List of data items
        output_path: Path to save converted data
    """
    
    ovis_format = {
        "conversations": []
    }
    
    for i, item in enumerate(input_data):
        # Convert to standard format
        if "conversations" not in item:
            # If no conversations field, create one
            conversation_item = {
                "id": item.get("id", f"converted_{i}"),
                "image": item.get("image", ""),
                "conversations": [
                    {
                        "from": "human",
                        "value": item.get("question", "<image>\nDescribe the image.")
                    },
                    {
                        "from": "gpt", 
                        "value": item.get("answer", "I can see an image.")
                    }
                ]
            }
        else:
            conversation_item = item
        
        ovis_format["conversations"].append(conversation_item)
    
    # Save converted data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ovis_format, f, ensure_ascii=False, indent=2)
    
    print(f"Converted data saved to {output_path}")


def split_data(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Split data into train/val/test sets
    
    Args:
        data_path: Path to data file
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios should sum to 1.0"
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data["conversations"]
    total = len(conversations)
    
    # Calculate split indices
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split data
    train_data = {"conversations": conversations[:train_end]}
    val_data = {"conversations": conversations[train_end:val_end]}
    test_data = {"conversations": conversations[val_end:]}
    
    # Save splits
    base_path = data_path.rsplit('.', 1)[0]
    
    train_path = f"{base_path}_train.json"
    val_path = f"{base_path}_val.json"
    test_path = f"{base_path}_test.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Data split completed:")
    print(f"  Train: {len(train_data['conversations'])} samples -> {train_path}")
    print(f"  Val: {len(val_data['conversations'])} samples -> {val_path}")
    print(f"  Test: {len(test_data['conversations'])} samples -> {test_path}")


def create_sample_data(output_path: str, num_samples: int = 10):
    """Create sample data for testing"""
    
    sample_data = {
        "conversations": []
    }
    
    sample_questions = [
        "What do you see in this image?",
        "Describe the scene in detail.",
        "What are the main objects in this image?", 
        "What is happening in this picture?",
        "Can you identify the colors in this image?",
        "What is the mood or atmosphere of this image?",
        "Are there any people in this image?",
        "What type of location is shown?",
        "What time of day does this appear to be?",
        "What would you title this image?"
    ]
    
    sample_answers = [
        "I can see an interesting image with various elements.",
        "This image shows a detailed scene with multiple components.",
        "The main objects include several distinct items.",
        "There appears to be activity happening in this scene.",
        "I can identify several different colors throughout the image.",
        "The image conveys a particular mood and atmosphere.",
        "I can observe people in various parts of the image.",
        "This appears to be taken in a specific type of location.",
        "Based on the lighting, this seems to be during a particular time.",
        "This image could be titled based on its main theme."
    ]
    
    for i in range(num_samples):
        item = {
            "id": f"sample_{i+1:03d}",
            "image": f"sample_image_{i+1:03d}.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{sample_questions[i % len(sample_questions)]}"
                },
                {
                    "from": "gpt",
                    "value": sample_answers[i % len(sample_answers)]
                }
            ]
        }
        sample_data["conversations"].append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created {num_samples} sample data entries in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Data preparation utilities for Ovis training")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data format')
    validate_parser.add_argument('data_path', help='Path to data file')
    validate_parser.add_argument('image_folder', help='Path to image folder')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split data into train/val/test')
    split_parser.add_argument('data_path', help='Path to data file')
    split_parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    split_parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample data')
    sample_parser.add_argument('output_path', help='Output path for sample data')
    sample_parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to create')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        is_valid, errors = validate_data_format(args.data_path, args.image_folder)
        if is_valid:
            print("✓ Data format is valid!")
        else:
            print("✗ Data format has errors:")
            for error in errors:
                print(f"  - {error}")
    
    elif args.command == 'split':
        split_data(args.data_path, args.train_ratio, args.val_ratio, args.test_ratio)
    
    elif args.command == 'sample':
        create_sample_data(args.output_path, args.num_samples)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()