"""
This code is alternative splitting script for tool-calling dataset only.
Re-split tool-calling dataset by IMAGE (not Q&A pairs)

This ensures no image overlap between train/val/test sets.
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def resplit_by_images(
    input_path: str,
    output_train: str,
    output_val: str,
    output_test: str = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset by images (no overlap)

    Args:
        input_path: Path to merged dataset JSON
        output_train: Output path for training set
        output_val: Output path for validation set
        output_test: Output path for test set (optional)
        train_ratio: Fraction of images for training (default 0.8)
        val_ratio: Fraction of images for validation (default 0.1)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Load data
    print(f"Loading dataset from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Group Q&A by image
    image_to_qa = defaultdict(list)
    for sample in data:
        image_to_qa[sample['image']].append(sample)

    unique_images = list(image_to_qa.keys())
    print(f"Unique images: {len(unique_images)}")
    print(f"Avg Q&A per image: {len(data)/len(unique_images):.2f}")

    # Shuffle images
    random.shuffle(unique_images)

    # Calculate split sizes
    n_images = len(unique_images)

    if output_test:
        # Three-way split: train + val + test
        test_ratio = 1.0 - train_ratio - val_ratio
        assert test_ratio >= 0, f"train_ratio + val_ratio must be <= 1.0"

        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        train_images = set(unique_images[:n_train])
        val_images = set(unique_images[n_train:n_train+n_val])
        test_images = set(unique_images[n_train+n_val:])
    else:
        # Two-way split: train + val only
        # Normalize ratios to sum to 1.0
        total_ratio = train_ratio + val_ratio
        train_ratio_norm = train_ratio / total_ratio
        val_ratio_norm = val_ratio / total_ratio

        n_train = int(n_images * train_ratio_norm)
        n_val = n_images - n_train  # Use all remaining for val

        train_images = set(unique_images[:n_train])
        val_images = set(unique_images[n_train:])
        test_images = set()

    # Assign Q&A based on image assignment
    train_qa = [sample for sample in data if sample['image'] in train_images]
    val_qa = [sample for sample in data if sample['image'] in val_images]
    test_qa = [sample for sample in data if sample['image'] in test_images]

    # Verify no overlap
    assert len(train_images & val_images) == 0, "Train/val image overlap!"
    assert len(train_images & test_images) == 0, "Train/test image overlap!"
    assert len(val_images & test_images) == 0, "Val/test image overlap!"

    # Print statistics
    print("\n=== Split Statistics ===")
    print(f"Train: {len(train_qa)} samples, {len(train_images)} images ({100*len(train_images)/n_images:.1f}%)")
    print(f"Val:   {len(val_qa)} samples, {len(val_images)} images ({100*len(val_images)/n_images:.1f}%)")
    if output_test:
        print(f"Test:  {len(test_qa)} samples, {len(test_images)} images ({100*len(test_images)/n_images:.1f}%)")

    print(f"\nTrain avg Q&A per image: {len(train_qa)/len(train_images):.2f}")
    print(f"Val avg Q&A per image:   {len(val_qa)/len(val_images):.2f}")
    if output_test:
        print(f"Test avg Q&A per image:  {len(test_qa)/len(test_images):.2f}")

    # Verify no overlap
    print(f"\n✅ Image overlap verification:")
    print(f"   Train ∩ Val: {len(train_images & val_images)} (should be 0)")
    if output_test:
        print(f"   Train ∩ Test: {len(train_images & test_images)} (should be 0)")
        print(f"   Val ∩ Test: {len(val_images & test_images)} (should be 0)")

    # Save splits
    print(f"\nSaving train set to {output_train}...")
    with open(output_train, 'w') as f:
        json.dump(train_qa, f, indent=2)

    print(f"Saving val set to {output_val}...")
    with open(output_val, 'w') as f:
        json.dump(val_qa, f, indent=2)

    if output_test:
        print(f"Saving test set to {output_test}...")
        with open(output_test, 'w') as f:
            json.dump(test_qa, f, indent=2)

    print("\n✅ Dataset re-split complete!")
    print(f"No image overlap between splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input merged dataset JSON')
    parser.add_argument('--output_train', required=True, help='Output train JSON')
    parser.add_argument('--output_val', required=True, help='Output val JSON')
    parser.add_argument('--output_test', default=None, help='Output test JSON (optional)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    resplit_by_images(
        input_path=args.input,
        output_train=args.output_train,
        output_val=args.output_val,
        output_test=args.output_test,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # python utils/resplit_dataset_by_images.py \
    #     --input /mnt/nas1/data/coco/refcoco_vlm_results_theo_ready_to_train/merged_toolcall_dataset.json \
    #     --output_train /mnt/nas1/data/coco/refcoco_vlm_results_theo_ready_to_train/train_toolcall_94.json \
    #     --output_val /mnt/nas1/data/coco/refcoco_vlm_results_theo_ready_to_train/val_toolcall_6.json \
    #     --train_ratio 0.94 \
    #     --val_ratio 0.06 \
    #     --seed 42
