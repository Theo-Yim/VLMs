#!/usr/bin/env python3
"""
Data Augmentation Script for House Defect Inspection Dataset
Augments images and creates new training samples while preserving defect visibility
"""

import json
import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import argparse
from tqdm import tqdm
import shutil


def augment_image_safe(image_path, output_path, aug_type="brightness"):
    """
    Apply safe augmentation that preserves defect visibility

    Args:
        image_path: Path to original image
        output_path: Path to save augmented image
        aug_type: Type of augmentation (brightness, rotation, flip, etc.)
    """
    img = Image.open(image_path)

    if aug_type == "brightness_up":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.15)  # 15% brighter

    elif aug_type == "brightness_down":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.85)  # 15% darker

    elif aug_type == "contrast_up":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)

    elif aug_type == "contrast_down":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.85)

    elif aug_type == "rotation_cw":
        img = img.rotate(-3, expand=False, fillcolor='white')

    elif aug_type == "rotation_ccw":
        img = img.rotate(3, expand=False, fillcolor='white')

    elif aug_type == "horizontal_flip":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    elif aug_type == "slight_blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    elif aug_type == "color_jitter":
        # Slight color variation (different camera sensors)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.95, 1.05))

    img.save(output_path, quality=95)
    return output_path


def create_augmented_sample(original_sample, image_filename_new, aug_type):
    """
    Create a new dataset sample with augmented image

    Args:
        original_sample: Original dataset entry
        image_filename_new: New augmented image filename
        aug_type: Type of augmentation applied
    """
    new_sample = original_sample.copy()
    new_sample['image'] = image_filename_new

    # Update ID to reflect augmentation
    original_id = new_sample.get('id', 'sample')
    new_sample['id'] = f"{original_id}_aug_{aug_type}"

    return new_sample


def augment_dataset(
    input_json,
    image_folder,
    output_json,
    output_image_folder,
    augmentation_factor=2,
    augmentation_types=None
):
    """
    Augment the entire dataset

    Args:
        input_json: Path to original dataset JSON
        image_folder: Folder containing original images
        output_json: Path to save augmented dataset JSON
        output_image_folder: Folder to save augmented images
        augmentation_factor: How many augmented versions per image (1-5 recommended)
        augmentation_types: List of augmentation types to apply (None = use defaults)
    """
    # Default safe augmentations for defect detection
    if augmentation_types is None:
        augmentation_types = [
            "brightness_up",
            "brightness_down",
            "contrast_up",
            "rotation_cw",
            "rotation_ccw",
        ]

    # Load original dataset
    print(f"Loading dataset from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Original dataset size: {len(dataset)}")

    # Create output folder
    os.makedirs(output_image_folder, exist_ok=True)

    # Augmented dataset will include original + augmented
    augmented_dataset = []

    # First, copy all original samples
    print("Copying original images...")
    for sample in tqdm(dataset, desc="Originals"):
        # Copy original image
        src_image = os.path.join(image_folder, sample['image'])
        if os.path.exists(src_image):
            dst_image = os.path.join(output_image_folder, sample['image'])
            shutil.copy2(src_image, dst_image)
            augmented_dataset.append(sample)
        else:
            print(f"Warning: Image not found: {src_image}")

    # Now create augmented versions
    print(f"\nCreating {augmentation_factor} augmented versions per image...")
    for sample in tqdm(dataset, desc="Augmenting"):
        src_image_path = os.path.join(image_folder, sample['image'])

        if not os.path.exists(src_image_path):
            continue

        # Create multiple augmented versions
        for i in range(augmentation_factor):
            # Randomly select augmentation type
            aug_type = random.choice(augmentation_types)

            # Generate new filename
            base_name = Path(sample['image']).stem
            ext = Path(sample['image']).suffix
            new_image_name = f"{base_name}_aug_{aug_type}_{i}{ext}"
            new_image_path = os.path.join(output_image_folder, new_image_name)

            # Apply augmentation
            try:
                augment_image_safe(src_image_path, new_image_path, aug_type)

                # Create new sample
                new_sample = create_augmented_sample(sample, new_image_name, aug_type)
                augmented_dataset.append(new_sample)

            except Exception as e:
                print(f"Error augmenting {sample['image']}: {e}")

    # Save augmented dataset
    print(f"\nSaving augmented dataset to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(augmented_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"AUGMENTATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Original dataset: {len(dataset)} samples")
    print(f"Augmented dataset: {len(augmented_dataset)} samples")
    print(f"Increase: {len(augmented_dataset) / len(dataset):.1f}x")
    print(f"Images saved to: {output_image_folder}")
    print(f"Dataset saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment house defect inspection dataset"
    )
    parser.add_argument(
        "--input_json",
        default="/workspace/VLMs/utils/lh-poc/dataset/dataset_lh_theo_v2_train.json",
        help="Path to input dataset JSON"
    )
    parser.add_argument(
        "--image_folder",
        default="/home/Theo-Yim/data/lh-poc/lh-data-image-train/",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output_json",
        default="/workspace/VLMs/utils/lh-poc/dataset/dataset_lh_theo_v2_train_augmented.json",
        help="Path to save augmented dataset JSON"
    )
    parser.add_argument(
        "--output_image_folder",
        default="/home/Theo-Yim/data/lh-poc/lh-data-image-train-augmented/",
        help="Path to save augmented images"
    )
    parser.add_argument(
        "--augmentation_factor",
        type=int,
        default=2,
        help="Number of augmented versions per image (1-5 recommended)"
    )

    args = parser.parse_args()

    augment_dataset(
        input_json=args.input_json,
        image_folder=args.image_folder,
        output_json=args.output_json,
        output_image_folder=args.output_image_folder,
        augmentation_factor=args.augmentation_factor
    )


if __name__ == "__main__":
    main()
