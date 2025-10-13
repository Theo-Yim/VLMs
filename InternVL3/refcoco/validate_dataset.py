#!/usr/bin/env python3
"""
Validate crop tool dataset quality and identify issues.

Usage:
    python validate_dataset.py --file_path /path/to/refcoco_qa_pairs.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def extract_crop_calls(text: str) -> List[str]:
    """Extract all crop tool calls from text."""
    if not text:
        return []

    # Pattern: <tool_call>Crop [x,y,x2,y2]</tool_call>
    pattern = r"<tool_call>Crop\s*\[[^\]]+\]</tool_call>"
    matches = re.findall(pattern, text)
    return matches


def has_crop_mentions(text: str) -> bool:
    """Check if text mentions crop tool in reasoning without calling it."""
    if not text:
        return False

    # Look for mentions of crop/zoom/inspect in <think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)

    for think_content in think_matches:
        # Check if mentions crop-related actions
        if re.search(
            r"\b(crop|zoom|inspect|examine|look\s+closer|use\s+the\s+.*\s+tool)\b",
            think_content,
            re.IGNORECASE,
        ):
            # But only if there's no actual tool call
            if "<tool_call>" not in think_content:
                return True
    return False


def check_easy_question(question: str) -> Tuple[bool, str]:
    """
    Conservatively identify very easy questions that clearly don't need cropping.
    Returns (is_easy, reason)
    """
    question_lower = question.lower()

    # Only mark as easy if EXTREMELY obvious
    very_easy_patterns = [
        (r"^how many (people|person)s? (are|is) (in|visible)", "counting people from full image"),
        (r"^what (color|type) is the (background|sky|floor|ground)", "background elements"),
    ]

    for pattern, reason in very_easy_patterns:
        if re.search(pattern, question_lower):
            return True, reason

    return False, ""


def analyze_dataset(file_path: str) -> Tuple[Dict, Dict]:
    """Generate comprehensive quality report."""

    stats = {
        "total_samples": 0,
        "total_qa_pairs": 0,
        "with_crop": 0,
        "without_crop": 0,
        "mentions_crop_without_call": 0,
        "easy_questions": 0,
        "missing_answer_tags": 0,
        "too_verbose": 0,
        "multi_crop": 0,
        "crop_counts": Counter(),
    }

    issues_by_type = defaultdict(list)

    # Load data
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # JSON array format
            samples = json.load(f)
        else:
            # JSONL format
            samples = [json.loads(line) for line in f if line.strip()]

    # Process each sample
    for line_num, sample in enumerate(samples, 1):
        try:
            stats["total_samples"] += 1

            # Handle both QnA format and conversation format
            qa_pairs = []
            if "QnA" in sample:
                # QnA format
                for qa in sample.get("QnA", []):
                    qa_pairs.append(
                        {"question": qa.get("Q", ""), "answer": qa.get("A3", qa.get("A", ""))}
                    )
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
                            if question and answer:  # Skip empty
                                qa_pairs.append({"question": question, "answer": answer})

            # Analyze each QA pair
            for qa in qa_pairs:
                stats["total_qa_pairs"] += 1
                answer = qa["answer"]
                question = qa["question"]

                # Count crops
                crop_calls = extract_crop_calls(answer)
                num_crops = len(crop_calls)
                stats["crop_counts"][num_crops] += 1

                if num_crops > 0:
                    stats["with_crop"] += 1
                    if num_crops > 1:
                        stats["multi_crop"] += 1
                else:
                    stats["without_crop"] += 1

                # Check issues
                if has_crop_mentions(answer) and num_crops == 0:
                    stats["mentions_crop_without_call"] += 1
                    issues_by_type["mentions_crop_without_call"].append(
                        {
                            "line": line_num,
                            "image_id": sample.get("image_id", "unknown"),
                            "question": question[:80],
                        }
                    )

                is_easy, reason = check_easy_question(question)
                if is_easy:
                    stats["easy_questions"] += 1
                    issues_by_type["easy_question"].append(
                        {
                            "line": line_num,
                            "image_id": sample.get("image_id", "unknown"),
                            "question": question[:80],
                            "reason": reason,
                        }
                    )

                if "<answer>" not in answer:
                    stats["missing_answer_tags"] += 1

                if len(answer) > 500:
                    stats["too_verbose"] += 1

        except Exception as e:
            print(f"Error processing sample {line_num}: {e}")
            continue

    return stats, issues_by_type


def print_report(stats: Dict, issues_by_type: Dict):
    """Print formatted quality report."""
    print("\n" + "=" * 80)
    print("DATASET QUALITY REPORT")
    print("=" * 80)

    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total QA pairs: {stats['total_qa_pairs']}")
    print(
        f"  Average QA pairs per sample: {stats['total_qa_pairs'] / max(stats['total_samples'], 1):.2f}"
    )

    print(f"\n🔧 CROP TOOL USAGE:")
    print(
        f"  With crop calls: {stats['with_crop']} ({stats['with_crop'] / max(stats['total_qa_pairs'], 1) * 100:.1f}%)"
    )
    print(
        f"  Without crop calls: {stats['without_crop']} ({stats['without_crop'] / max(stats['total_qa_pairs'], 1) * 100:.1f}%)"
    )
    print(f"  Multi-crop (>1): {stats['multi_crop']}")

    print(f"\n  Crop count distribution:")
    for num_crops in sorted(stats["crop_counts"].keys()):
        count = stats["crop_counts"][num_crops]
        pct = count / max(stats["total_qa_pairs"], 1) * 100
        print(f"    {num_crops} crops: {count} ({pct:.1f}%)")

    print(f"\n⚠️  ISSUES FOUND:")
    print(f"  Mentions crop without calling: {stats['mentions_crop_without_call']}")
    print(f"  Easy questions: {stats['easy_questions']}")
    print(f"  Missing answer tags: {stats['missing_answer_tags']}")
    print(f"  Too verbose (>500 chars): {stats['too_verbose']}")

    print(f"\n📋 DETAILED ISSUES BY TYPE:")
    for issue_type, issue_list in issues_by_type.items():
        print(f"\n  {issue_type.upper()}: {len(issue_list)} occurrences")
        # Show first 5 examples
        for i, issue in enumerate(issue_list[:5], 1):
            print(f"    {i}. Line {issue['line']} (image {issue['image_id']})")
            if "question" in issue:
                print(f"       Q: {issue['question']}")
            if "reason" in issue:
                print(f"       Reason: {issue['reason']}")

    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    if stats["without_crop"] > stats["total_qa_pairs"] * 0.1:
        print(
            f"  ⚠️  HIGH: {stats['without_crop']} samples without crop calls ({stats['without_crop'] / stats['total_qa_pairs'] * 100:.1f}%)"
        )
        print(f"      → Consider filtering these out to maintain consistent tool-use training")

    if stats["mentions_crop_without_call"] > 0:
        print(f"  ⚠️  HIGH: {stats['mentions_crop_without_call']} inconsistent crop mentions")
        print(f"      → Fix LLM prompts to enforce tool calling consistency")

    if stats["easy_questions"] > 0:
        print(f"  ⚠️  MEDIUM: {stats['easy_questions']} easy questions detected")
        print(f"      → Consider filtering these at generation time")

    if stats["too_verbose"] > stats["total_qa_pairs"] * 0.2:
        print(f"  ⚠️  MEDIUM: {stats['too_verbose']} verbose answers")
        print(f"      → Improve prompts for conciseness")

    print(f"\n✅ QUALITY SCORE: {calculate_quality_score(stats):.1f}/100")


def calculate_quality_score(stats: Dict) -> float:
    """Calculate overall quality score (0-100)."""
    score = 100.0

    # Deduct for issues
    if stats["total_qa_pairs"] > 0:
        score -= (stats["without_crop"] / stats["total_qa_pairs"]) * 30  # 30 points for crop usage
        score -= (
            stats["mentions_crop_without_call"] / stats["total_qa_pairs"]
        ) * 20  # 20 points for consistency
        score -= (
            stats["easy_questions"] / stats["total_qa_pairs"]
        ) * 15  # 15 points for question quality
        score -= (
            stats["missing_answer_tags"] / stats["total_qa_pairs"]
        ) * 10  # 10 points for format
        score -= (stats["too_verbose"] / stats["total_qa_pairs"]) * 10  # 10 points for conciseness

    return max(0.0, score)


def main():
    parser = argparse.ArgumentParser(description="Validate crop tool dataset quality")
    parser.add_argument(
        "--file_path",
        type=str,
        default="/mnt/nas3/Data/coco/refcoco_vlm_results_theo_ready_to_train/refcoco_qa_pairs_croptool.json",
        help="Path to JSON/JSONL file to validate",
    )
    parser.add_argument(
        "--output_issues", type=str, help="Optional: Save detailed issues to JSON file"
    )

    args = parser.parse_args()

    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        return

    print(f"Analyzing: {args.file_path}")
    stats, issues_by_type = analyze_dataset(args.file_path)
    print_report(stats, issues_by_type)

    if args.output_issues:
        with open(args.output_issues, "w", encoding="utf-8") as f:
            json.dump(dict(issues_by_type), f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed issues saved to: {args.output_issues}")


if __name__ == "__main__":
    main()
