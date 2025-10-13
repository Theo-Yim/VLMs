"""
Stage 1: Improved Raw Identity Dataset Generation

Generates raw Q&A data with:
- 1 mock name per person
- 1 question per person
- 1 answer with tool call placeholders

Usage:
    CUDA_VISIBLE_DEVICES=1 python identity_stage1.py --start 0 --end 1 --output_folder=InternVL3/refcoco_id/stage1_test
"""

import argparse
import os

from tqdm import tqdm

from InternVL3.refcoco_id.processor_stage1 import IdentityStage1Processor


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Generate improved raw identity Q&A dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="LLM model path",
    )
    parser.add_argument(
        "--merged_data",
        type=str,
        default="/workspace/VLMs/merged_refcoco_data.pkl",
        help="Path to merged RefCOCO data",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default="/mnt/nas3/Data/coco",
        help="Path to COCO dataset",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for processing",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index for processing (-1 for all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Custom output folder (default: auto-detect from coco_path)",
    )
    parser.add_argument(
        "--multi_person",
        action="store_true",
        help="Generate multi-person Q&A (requires 2+ people per image)",
    )

    args = parser.parse_args()

    # Initialize processor
    print("=" * 80)
    if args.multi_person:
        print("STAGE 1: Multi-Person Identity Dataset Generation")
    else:
        print("STAGE 1: Improved Raw Identity Dataset Generation")
    print("=" * 80)
    processor = IdentityStage1Processor(
        model_path=args.model, output_path=args.coco_path, output_folder=args.output_folder
    )

    # Load datasets
    data_list = processor.load_datasets(args.merged_data)

    # Filter for multi-person images if needed
    if args.multi_person:
        data_list = [entry for entry in data_list if len(entry["annotations"]) >= 2]
        print(f"Filtered to {len(data_list)} images with 2+ people")

    # Apply range
    if args.end == -1:
        args.end = len(data_list)
    data_list = data_list[args.start : args.end]

    print(f"\nProcessing {len(data_list)} images (index {args.start} to {args.end})")
    print("=" * 80)

    # Statistics
    total_images = 0
    total_people = 0
    total_qna = 0
    total_llm_calls = 0
    total_errors = 0

    # Process each image
    for idx, data_entry in enumerate(tqdm(data_list, desc="Processing images")):
        image_id = data_entry["image_id"]
        output_file = os.path.join(processor.output_folder, f"{image_id}.json")

        # Skip if already processed (resume mode)
        if args.resume and os.path.exists(output_file):
            print(f"Skipping {image_id} (already processed)")
            continue

        try:
            num_people = len(data_entry["annotations"])

            print(f"\n{'=' * 80}")
            print(f"Image {idx + 1}/{len(data_list)}: {image_id}")
            print(f"People: {num_people}")

            # Step 1: Generate mock names (1 per person, unique)
            print(f"  [1/3] Generating {num_people} unique mock names...")
            names = processor.assign_names_to_people(data_entry)
            print(f"    Names: {', '.join(names)}")
            total_llm_calls += num_people

            # Step 2: Generate single-person questions and answers (1 per person)
            if not args.multi_person:
                print("  [2/3] Generating diverse questions and answers with RICH reasoning...")
                num_calls = processor.generate_questions_and_answers(data_entry)
                num_qna = len(data_entry["QnA"])
                print(f"    Generated {num_qna} single-person Q&A pairs ({num_calls} LLM calls)")
                total_llm_calls += num_calls
            else:
                # Skip single-person Q&A if multi-person mode
                data_entry["QnA"] = []
                print("  [2/3] Skipping single-person Q&A (multi-person mode)")

            # Step 3: Generate multi-person questions and answers (optional)
            if args.multi_person:
                print("  [3/3] Generating multi-person Q&A pairs...")
                num_calls = processor.generate_multi_person_questions_and_answers(data_entry)
                num_qna_multi = len(data_entry.get("QnA_multi", []))
                print(f"    Generated {num_qna_multi} multi-person Q&A pairs ({num_calls} LLM calls)")
                total_llm_calls += num_calls
                num_qna = num_qna_multi
            else:
                num_qna = len(data_entry.get("QnA", []))

            # Save results immediately
            processor.save_results(data_entry, output_file)

            # Update stats
            total_images += 1
            total_people += num_people
            total_qna += num_qna

        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            import traceback

            traceback.print_exc()
            total_errors += 1
            continue

    # Final statistics
    print("\n" + "=" * 80)
    print("STAGE 1 COMPLETE")
    print("=" * 80)
    print(f"Images processed: {total_images}")
    print(f"Total people: {total_people}")
    print(f"Total Q&A pairs: {total_qna}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Errors: {total_errors}")
    print(f"Output directory: {processor.output_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()
