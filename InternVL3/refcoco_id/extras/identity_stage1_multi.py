"""
Stage 1 Multi: Multi-Person Identity Dataset Generation

Generates multi-person Q&A data where each question requires identifying 2+ people.

Usage:
    CUDA_VISIBLE_DEVICES=1 python identity_stage1_multi.py \
        --existing_stage1 /mnt/nas3/Data/coco/refcoco_identity_stage1 \
        --start 0 --end 100
"""

import argparse
import os

from tqdm import tqdm

from InternVL3.refcoco_id.processor_stage1_multi import IdentityStage1MultiProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 Multi: Generate multi-person identity Q&A dataset"
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
        "--existing_stage1",
        type=str,
        default=None,
        help="Path to existing Stage 1 folder (to reuse person names)",
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

    args = parser.parse_args()

    # Initialize processor
    print("=" * 80)
    print("STAGE 1 MULTI: Multi-Person Identity Dataset Generation")
    print("=" * 80)
    processor = IdentityStage1MultiProcessor(
        model_path=args.model,
        output_path=args.coco_path,
        output_folder=args.output_folder,
        existing_stage1_folder=args.existing_stage1,
    )

    # Load datasets (only images with 2+ people)
    data_list = processor.load_datasets(args.merged_data)

    # Apply range
    if args.end == -1:
        args.end = len(data_list)
    data_list = data_list[args.start : args.end]

    print(f"\nProcessing {len(data_list)} images (index {args.start} to {args.end})")
    print("=" * 80)

    # Statistics
    total_images = 0
    total_qna = 0
    total_llm_calls = 0
    total_errors = 0
    images_by_people_count = {}

    # Process each image
    for idx, data_entry in enumerate(tqdm(data_list, desc="Processing multi-person images")):
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

            # Track distribution
            if num_people not in images_by_people_count:
                images_by_people_count[num_people] = 0
            images_by_people_count[num_people] += 1

            # Load existing names from Stage 1 (or generate new ones)
            if args.existing_stage1:
                existing_names = processor.load_existing_names(image_id)
                if existing_names:
                    data_entry["person_names"] = existing_names
                    print(f"  Loaded existing names: {', '.join(existing_names)}")
                else:
                    print(f"  Warning: No existing names found for {image_id}, skipping...")
                    total_errors += 1
                    continue
            else:
                print(f"  Error: --existing_stage1 is required to load person names")
                total_errors += 1
                continue

            # Generate multi-person questions and answers
            print("  Generating multi-person Q&A pairs...")
            num_calls = processor.generate_multi_person_questions_and_answers(data_entry)
            num_qna = len(data_entry.get("QnA_multi", []))
            print(f"    Generated {num_qna} multi-person Q&A pairs ({num_calls} LLM calls)")
            total_llm_calls += num_calls

            # Save results immediately
            processor.save_results(data_entry, output_file)

            # Update stats
            total_images += 1
            total_qna += num_qna

        except Exception as e:
            print(f"\nError processing {image_id}: {e}")
            import traceback

            traceback.print_exc()
            total_errors += 1
            continue

    # Final statistics
    print("\n" + "=" * 80)
    print("STAGE 1 MULTI COMPLETE")
    print("=" * 80)
    print(f"Images processed: {total_images}")
    print(f"Total multi-person Q&A pairs: {total_qna}")
    print(f"Avg Q&A per image: {total_qna / total_images if total_images > 0 else 0:.2f}")
    print(f"Total LLM calls: {total_llm_calls}")
    print(f"Errors: {total_errors}")
    print(f"\nDistribution by number of people:")
    for num_people in sorted(images_by_people_count.keys()):
        print(f"  {num_people} people: {images_by_people_count[num_people]} images")
    print(f"Output directory: {processor.output_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()
