"""
Merge Single-Person and Multi-Person Identity Datasets

Combines the existing 31k single-person dataset with newly generated
multi-person dataset to create a comprehensive training set.

Usage:
    python merge_datasets.py \
        --single_person dataset/identity_qa_pairs_31k.json \
        --multi_person dataset/identity_qa_pairs_multi_14k.json \
        --output dataset/identity_qa_pairs_45k_merged.json
"""

import argparse
import json
from collections import defaultdict


def load_json(file_path: str) -> list:
    """Load JSON file"""
    print(f"Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} conversations")
    return data


def analyze_dataset(data: list, name: str):
    """Analyze and print dataset statistics"""
    print(f"\n{name} Statistics:")
    print("=" * 60)

    # Count by image
    image_counts = defaultdict(int)
    for entry in data:
        image_counts[entry["image_id"]] += 1

    print(f"Total conversations: {len(data)}")
    print(f"Unique images: {len(image_counts)}")
    print(f"Avg Q&A per image: {len(data) / len(image_counts):.2f}")

    # Count tool calls
    total_tool_calls = 0
    multi_tool_calls = 0
    for entry in data:
        gpt_value = entry["conversations"][-1]["value"]
        num_tool_calls = gpt_value.count("<tool_call>")
        total_tool_calls += num_tool_calls
        if num_tool_calls >= 2:
            multi_tool_calls += 1

    print(f"Total tool calls: {total_tool_calls}")
    print(f"Avg tool calls per Q&A: {total_tool_calls / len(data):.2f}")
    print(f"Q&A with 2+ tool calls: {multi_tool_calls} ({multi_tool_calls / len(data) * 100:.1f}%)")

    # Question length distribution
    question_lengths = []
    for entry in data:
        question = entry["conversations"][1]["value"].replace("<image>\n", "")
        question_lengths.append(len(question.split()))

    avg_q_len = sum(question_lengths) / len(question_lengths)
    print(f"Avg question length: {avg_q_len:.1f} words")

    # Answer length distribution
    answer_lengths = []
    for entry in data:
        answer = entry["conversations"][-1]["value"]
        answer_lengths.append(len(answer.split()))

    avg_a_len = sum(answer_lengths) / len(answer_lengths)
    print(f"Avg answer length: {avg_a_len:.1f} words")


def merge_datasets(single_person_data: list, multi_person_data: list) -> list:
    """
    Merge single-person and multi-person datasets

    Strategy:
    - Keep all single-person Q&A
    - Add all multi-person Q&A
    - Sort by image_id for consistency
    """
    print("\nMerging datasets...")

    # Combine all conversations
    merged = single_person_data + multi_person_data

    # Sort by image_id for consistency
    merged.sort(key=lambda x: x["image_id"])

    print(f"  Total merged conversations: {len(merged)}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge identity datasets")
    parser.add_argument(
        "--single_person",
        type=str,
        required=True,
        help="Path to single-person dataset (31k)",
    )
    parser.add_argument(
        "--multi_person",
        type=str,
        required=True,
        help="Path to multi-person dataset (14k)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged dataset",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze datasets before merging",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("IDENTITY DATASET MERGER")
    print("=" * 80)

    # Load datasets
    single_person_data = load_json(args.single_person)
    multi_person_data = load_json(args.multi_person)

    # Analyze if requested
    if args.analyze:
        analyze_dataset(single_person_data, "Single-Person Dataset")
        analyze_dataset(multi_person_data, "Multi-Person Dataset")

    # Merge
    merged_data = merge_datasets(single_person_data, multi_person_data)

    # Analyze merged dataset
    analyze_dataset(merged_data, "Merged Dataset")

    # Save
    print(f"\nSaving merged dataset to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(merged_data)} conversations to {args.output}")

    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
