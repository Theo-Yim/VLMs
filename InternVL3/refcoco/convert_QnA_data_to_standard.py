#!/usr/bin/env python3
"""
Standalone script to convert QnA format data to standard conversation format.
Usage: python convert_data.py input.jsonl output.json [image_base_path]
"""

import argparse
import json
import re
from typing import Any, Dict


def parse_and_replace_tool_calls(text: str) -> str:
    """
    Parse tool calls in the format {Crop person 1 [0.00, 141.43, 79.23, 480.00]}
    and replace with <tool_call>Crop [0.00, 141.43, 79.23, 480.00]</tool_call>

    Example Usage:
    When a3_answer is "<think>Let me closely look at the person.\n\n{Crop person 2 [181, 16, 220, 191]}\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"
    a3_answer_processed = parse_and_replace_tool_calls(a3_answer)
    a3_answer_processed is "<think>Let me closely look at the person.\n\n<tool_call>Crop [181, 16, 220, 191]</tool_call>\n\nUpon closer inspection, the person is engaging with the camera.</think>\n\n<answer>The person is engaging with the camera.\n</answer>"
    messages.append({"role": "assistant", "content": a3_answer_processed})
    """
    pattern = r"\{Crop[^}]*\[([\d.,\s]+)\]\}"

    def replace_func(match):
        coords = match.group(1)
        return f"<tool_call>Crop [{coords}]</tool_call>"

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


def process_qna_sample(sample: Dict[str, Any], image_base_path: str = "") -> Dict[str, Any]:
    """
    Convert QnA format sample to standard conversation format

    Args:
        sample: QnA format sample with 'QnA' field
        image_base_path: Base path for resolving relative image paths

    Returns:
        Sample in standard conversation format
    """
    # Convert QnA pairs to conversations
    conversations = []
    for qa in sample.get("QnA", []):
        question = qa.get("Q", "")
        answer = qa.get("A3", qa.get("A", ""))

        # User message with image placeholder
        conversations.append({"from": "human", "value": f"<image>\n{question}"})

        # Assistant message - process tool calls if present
        if answer and has_tool_calls(answer):
            # Convert {Crop ...} to <tool_call>Crop ...</tool_call>
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
    input_jsonl_path: str, output_json_path: str, image_base_path: str = ""
):
    """
    Convert JSONL file with QnA format to JSON with standard conversation format

    Args:
        input_jsonl_path: Path to input JSONL file
        output_json_path: Path to output JSON file
        image_base_path: Base path for resolving image paths
    """
    converted_samples = []

    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    format_type = detect_format(sample)

                    if format_type == "qna":
                        converted_sample = process_qna_sample(sample, image_base_path)
                        converted_samples.append(converted_sample)
                    elif format_type == "conversation":
                        # Already in standard format
                        converted_samples.append(sample)
                    else:
                        print(f"Warning: Unknown format in line {line_num}, skipping")

                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")

    # Save converted samples
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)

    print(
        f"Converted {len(converted_samples)} samples from {input_jsonl_path} to {output_json_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert QnA format to standard conversation format"
    )
    parser.add_argument("input_file", help="Input JSONL file with QnA format")
    parser.add_argument("output_file", help="Output JSON file with conversation format")
    parser.add_argument("--image_base_path", default="", help="Base path for resolving image paths")

    args = parser.parse_args()

    convert_jsonl_to_standard(
        input_jsonl_path=args.input_file,
        output_json_path=args.output_file,
        image_base_path=args.image_base_path,
    )

    print(f"Conversion completed: {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    main()
