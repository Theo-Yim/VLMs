#!/usr/bin/env python3
"""
Create JSONL file from multiple JSON files containing Q&A pairs.
Each Q&A pair becomes a separate JSON object in the output JSONL file.

Usage:
    python create_jsonl.py --input_dir /path/to/json/files --output_file output.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def process_json_file(json_file_path: Path) -> List[Dict[str, Any]]:
    """
    Process a single JSON file and create separate entries for each Q&A pair.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        List of JSON objects, one for each Q&A pair
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract common fields
        image_path = data.get("image_path", "")
        image_id = data.get("image_id", "")
        qna_pairs = data.get("QnA", [])
        
        # Create separate entries for each Q&A pair
        entries = []
        for qa_pair in qna_pairs:
            # Only extract Q and A3 fields, filtering out any other data
            filtered_qa = {
                "Q": qa_pair.get("Q", ""),
                "A3": qa_pair.get("A3", "")
            }
            entry = {
                "image_path": image_path,
                "image_id": image_id,
                "QnA": [filtered_qa]  # Wrap single filtered Q&A pair in array
            }
            entries.append(entry)
        
        return entries
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing {json_file_path}: {e}")
        return []


def create_jsonl_from_directory(input_dir: Path, output_file: Path) -> None:
    """
    Process all JSON files in input directory and create JSONL output.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Output JSONL file path
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process all JSON files and collect entries
    all_entries = []
    processed_files = 0
    total_qa_pairs = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        entries = process_json_file(json_file)
        all_entries.extend(entries)
        processed_files += 1
        total_qa_pairs += len(entries)
    
    # Write to JSONL file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print("\nSummary:")
    print(f"- Processed {processed_files} JSON files")
    print(f"- Generated {total_qa_pairs} Q&A entries")
    print(f"- Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JSONL file from multiple JSON files containing Q&A pairs"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/nas3/Data/coco/refcoco_vlm_results_theo_llm/",
        help="Input directory containing JSON files (default: /mnt/nas3/Data/coco/refcoco_vlm_results_theo_llm/)"
    )
    parser.add_argument(
        "--output_file", 
        type=str,
        default="refcoco_qa_pairs.jsonl",
        help="Output JSONL file path (default: refcoco_qa_pairs.jsonl)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview of first few entries without writing output file"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    if args.preview:
        # Preview mode - show first few entries
        json_files = list(input_dir.glob("*.json"))[:2]  # Just first 2 files
        print("Preview mode - showing first few entries:\n")
        
        for json_file in json_files:
            entries = process_json_file(json_file)
            print(f"File: {json_file.name}")
            for i, entry in enumerate(entries[:2]):  # Show first 2 entries per file
                print(f"Entry {i+1}:")
                print(json.dumps(entry, indent=2, ensure_ascii=False))
                print()
        
        print("Preview complete. Use without --preview to generate full JSONL file.")
    else:
        # Full processing mode
        create_jsonl_from_directory(input_dir, output_file)


if __name__ == "__main__":
    main()
