#!/usr/bin/env python3
"""
Standalone script to convert QnA format data to standard conversation format.
Usage: python convert_data.py input.jsonl output.json
"""

import argparse
import json
import re
from typing import Any, Dict

# System prompt for crop tool instructions
SYSTEM_PROMPT = """You have access to a Crop tool for detailed visual analysis. When you need to examine a specific region of the image more closely, use the tool in this format: <tool_call>Crop [x, y, x2, y2]</tool_call>

After each tool use, a cropped image will be provided for your closer inspection. Use this capability to provide detailed and accurate responses based on visual evidence."""


def parse_and_replace_tool_calls(text: str) -> str:
    """
    Parse tool calls in the format {Crop person 1 [0.00, 141.43, 79.23, 480.00]}
    and replace with <tool_call>Crop [0.00, 141.43, 79.23, 480.00]</tool_call><image>

    Example Usage:
    When a3_answer is "<think>Let me closely look at the person.\n\n{Crop person 2 [181, 16, 220, 191]}\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"
    a3_answer_processed = parse_and_replace_tool_calls(a3_answer)
    a3_answer_processed is "<think>Let me closely look at the person.\n\n<tool_call>Crop [181, 16, 220, 191]</tool_call><image>\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"
    messages.append({"role": "assistant", "content": a3_answer_processed})
    """
    pattern = r"\{Crop[^}]*\[([\d.,\s]+)\]\}"

    def replace_func(match):
        coords = match.group(1)
        return f"<tool_call>Crop [{coords}]</tool_call><image>"

    return re.sub(pattern, replace_func, text)


def detect_format(sample: Dict[str, Any]) -> str:
    """Detect the format of a data sample"""
    if "QnA" in sample and isinstance(sample["QnA"], list):
        return "qna"
    elif "conversations" in sample and isinstance(sample["conversations"], list):
        return "conversation"
    else:
        return "unknown"


def has_tool_calls(text: str) -> bool:
    """Check if text contains tool calls"""
    return "{Crop" in text if text else False


def has_converted_tool_calls(text: str) -> bool:
    """Check if text contains converted tool calls"""
    return "<tool_call>Crop" in text if text else False


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
            if "<tool_call>" not in think_content and "{Crop" not in think_content:
                return True
    return False


def is_very_easy_question(question: str) -> bool:
    """
    Conservatively identify VERY easy questions that clearly don't need cropping.
    Only removes extremely obvious cases to be safe.
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


def should_keep_sample(sample: Dict[str, Any], filter_quality: bool = True) -> tuple:
    """
    Determine if sample should be kept based on quality criteria.
    Returns (should_keep, reason_if_filtered)
    """
    if not filter_quality:
        return True, ""

    # Check QnA format
    for qa in sample.get("QnA", []):
        question = qa.get("Q", "")
        answer = qa.get("A3", qa.get("A", ""))

        # Filter 1: Must have crop calls
        if not has_tool_calls(answer):
            return False, "no_crop_call"

        # Filter 2: Must not mention crop without calling
        if has_crop_mentions_without_call(answer):
            return False, "mentions_crop_without_call"

        # Filter 3: Remove very easy questions (conservative)
        if is_very_easy_question(question):
            return False, "very_easy_question"

    return True, ""


def process_qna_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert QnA format sample to standard conversation format with system prompt

    Args:
        sample: QnA format sample with 'QnA' field

    Returns:
        Sample in standard conversation format
    """
    # Add system prompt first
    conversations = [{"from": "system", "value": SYSTEM_PROMPT}]

    # Convert QnA pairs to conversations
    for qa in sample.get("QnA", []):
        question = qa.get("Q", "")
        answer = qa.get("A3", qa.get("A", ""))

        # User message with image placeholder
        conversations.append({"from": "human", "value": f"<image>\n{question}"})

        # Assistant message - process tool calls if present
        if answer and has_tool_calls(answer):
            # Convert {Crop ...} to <tool_call>Crop ...</tool_call><image>
            processed_answer = parse_and_replace_tool_calls(answer)
            conversations.append({"from": "gpt", "value": processed_answer})
        else:
            conversations.append({"from": "gpt", "value": answer})

    # Return in standard format
    result = {"conversations": conversations}

    # Copy over other fields
    image_path = sample.get("image_path", sample.get("image"))
    if image_path:
        result["image"] = image_path
    if "image_id" in sample:
        result["image_id"] = sample["image_id"]

    return result


def convert_jsonl_to_standard(
    input_jsonl_path: str, output_json_path: str, filter_quality: bool = True
):
    """
    Convert JSONL file with QnA format to JSON with standard conversation format

    Args:
        input_jsonl_path: Path to input JSONL file
        output_json_path: Path to output JSON file
        filter_quality: Apply quality filtering (default: True)
    """
    converted_samples = []
    stats = {
        "total": 0,
        "kept": 0,
        "filtered_no_crop": 0,
        "filtered_inconsistent": 0,
        "filtered_easy": 0,
    }

    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    stats["total"] += 1
                    format_type = detect_format(sample)

                    if format_type == "qna":
                        # Check quality before converting
                        should_keep, reason = should_keep_sample(sample, filter_quality)

                        if not should_keep:
                            if reason == "no_crop_call":
                                stats["filtered_no_crop"] += 1
                            elif reason == "mentions_crop_without_call":
                                stats["filtered_inconsistent"] += 1
                            elif reason == "very_easy_question":
                                stats["filtered_easy"] += 1
                            continue

                        converted_sample = process_qna_sample(sample)
                        converted_samples.append(converted_sample)
                        stats["kept"] += 1
                    elif format_type == "conversation":
                        # Already in standard format
                        converted_samples.append(sample)
                        stats["kept"] += 1
                    else:
                        print(f"Warning: Unknown format in line {line_num}, skipping")

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")

    # Save converted samples
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)

    # Print statistics
    print(f"\n{'=' * 80}")
    print(f"Conversion completed: {input_jsonl_path} -> {output_json_path}")
    print(f"{'=' * 80}")
    print(f"Total samples: {stats['total']}")
    print(f"Kept: {stats['kept']} ({stats['kept'] / max(stats['total'], 1) * 100:.1f}%)")

    if filter_quality:
        filtered = stats["total"] - stats["kept"]
        print(f"Filtered: {filtered} ({filtered / max(stats['total'], 1) * 100:.1f}%)")
        print(f"  - No crop call: {stats['filtered_no_crop']}")
        print(f"  - Inconsistent mentions: {stats['filtered_inconsistent']}")
        print(f"  - Very easy questions: {stats['filtered_easy']}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert QnA format to standard conversation format with optional quality filtering"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="refcoco_qa_pairs.jsonl",
        help="Input JSONL file with QnA format",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="refcoco_qa_pairs.json",
        help="Output JSON file with conversation format",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable quality filtering (keep all samples including those without crop calls)",
    )

    args = parser.parse_args()

    convert_jsonl_to_standard(
        input_jsonl_path=args.input_file,
        output_json_path=args.output_file,
        filter_quality=not args.no_filter,
    )


if __name__ == "__main__":
    main()
