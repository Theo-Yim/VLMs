#!/usr/bin/env python3
"""
Filter crop tool dataset to keep only high-quality samples.

Filtering criteria:
1. Must have at least one crop tool call
2. Must not mention crop without actually calling it
3. (Optional) Remove extremely easy questions

Usage:
    python filter_dataset.py --input_file input.json --output_file filtered.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def extract_crop_calls(text: str) -> List[str]:
    """Extract all crop tool calls from text."""
    if not text:
        return []
    pattern = r"<tool_call>Crop\s*\[[^\]]+\]</tool_call>"
    return re.findall(pattern, text)


def has_crop_mentions_without_call(text: str) -> bool:
    """Check if text mentions crop tool in reasoning without calling it."""
    if not text:
        return False

    # Look for mentions of crop/zoom/inspect in <think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)

    for think_content in think_matches:
        # Check if mentions crop-related actions
        if re.search(
            r"\b(crop|zoom|inspect|examine|look\s+closer|use\s+the\s+(crop\s+)?tool)\b",
            think_content,
            re.IGNORECASE,
        ):
            # But only if there's no actual tool call in that think block
            if "<tool_call>" not in think_content:
                return True
    return False


def is_very_easy_question(question: str) -> bool:
    """
    Conservatively identify VERY easy questions that clearly don't need cropping.
    Only removes extremely obvious cases.
    """
    question_lower = question.lower()

    # Only mark as easy if EXTREMELY obvious
    very_easy_patterns = [
        r"^how many (people|person)s? (are|is) (in|visible)",
        r"^what (color|type) is the (background|sky|floor|ground)\b",
    ]

    for pattern in very_easy_patterns:
        if re.search(pattern, question_lower):
            return True

    return False


def should_keep_sample(sample: Dict, filter_easy: bool = True) -> tuple[bool, str]:
    """
    Determine if sample should be kept based on filtering criteria.
    Returns (should_keep, reason_if_filtered)
    """
    # Handle both QnA format and conversation format
    qa_pairs = []

    if "QnA" in sample:
        # QnA format
        for qa in sample.get("QnA", []):
            qa_pairs.append({"question": qa.get("Q", ""), "answer": qa.get("A3", qa.get("A", ""))})
    elif "conversations" in sample:
        # Conversation format
        conversations = sample.get("conversations", [])
        for i in range(len(conversations)):
            if conversations[i].get("from") == "human":
                question = (
                    conversations[i]
                    .get("value", "")
                    .replace("<image>\n", "")
                    .replace("<image>", "")
                    .strip()
                )
                if i + 1 < len(conversations) and conversations[i + 1].get("from") == "gpt":
                    answer = conversations[i + 1].get("value", "")
                    if question and answer:
                        qa_pairs.append({"question": question, "answer": answer})

    # Check each QA pair
    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]

        # Filter 1: Must have crop calls
        crop_calls = extract_crop_calls(answer)
        if len(crop_calls) == 0:
            return False, "no_crop_call"

        # Filter 2: Must not mention crop without calling
        if has_crop_mentions_without_call(answer):
            return False, "mentions_crop_without_call"

        # Filter 3: Remove very easy questions (optional, conservative)
        if filter_easy and is_very_easy_question(question):
            return False, "very_easy_question"

    return True, ""


def filter_dataset(input_file: str, output_file: str, filter_easy: bool = True) -> Dict:
    """
    Filter dataset and save high-quality samples.
    Returns statistics about filtering.
    """
    # Load data
    with open(input_file, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            samples = json.load(f)
        else:
            samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} samples from {input_file}")

    # Filter samples
    filtered_samples = []
    stats = {
        "total": len(samples),
        "kept": 0,
        "filtered_no_crop_call": 0,
        "filtered_mentions_without_call": 0,
        "filtered_very_easy": 0,
    }

    for sample in samples:
        should_keep, reason = should_keep_sample(sample, filter_easy)

        if should_keep:
            filtered_samples.append(sample)
            stats["kept"] += 1
        else:
            if reason == "no_crop_call":
                stats["filtered_no_crop_call"] += 1
            elif reason == "mentions_crop_without_call":
                stats["filtered_mentions_without_call"] += 1
            elif reason == "very_easy_question":
                stats["filtered_very_easy"] += 1

    # Save filtered dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_samples, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print("FILTERING RESULTS")
    print(f"{'=' * 80}")
    print(f"Total samples: {stats['total']}")
    print(f"Kept: {stats['kept']} ({stats['kept'] / stats['total'] * 100:.1f}%)")
    print(
        f"Filtered out: {stats['total'] - stats['kept']} ({(stats['total'] - stats['kept']) / stats['total'] * 100:.1f}%)"
    )
    print(f"\nFiltering breakdown:")
    print(f"  No crop call: {stats['filtered_no_crop_call']}")
    print(f"  Mentions crop without call: {stats['filtered_mentions_without_call']}")
    print(f"  Very easy questions: {stats['filtered_very_easy']}")
    print(f"\nOutput saved to: {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter crop tool dataset for quality")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/mnt/nas3/Data/coco/refcoco_vlm_results_theo_ready_to_train/refcoco_qa_pairs_croptool.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/mnt/nas3/Data/coco/refcoco_vlm_results_theo_ready_to_train/refcoco_qa_pairs_croptool_filtered.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--keep_easy",
        action="store_true",
        help="Keep easy questions (default: filter them out conservatively)",
    )

    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    filter_easy = not args.keep_easy
    stats = filter_dataset(args.input_file, args.output_file, filter_easy)

    print(f"\n✅ Filtering complete!")
    print(f"Quality improvement: {stats['kept']} high-quality samples ready for training")


if __name__ == "__main__":
    main()
