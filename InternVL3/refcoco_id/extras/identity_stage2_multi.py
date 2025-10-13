"""
Stage 2 Multi: Refine and Convert Multi-Person Identity Dataset

Processes Stage 1 Multi outputs and converts to final training format.

Usage:
    python identity_stage2_multi.py \
        --stage1_multi_folder InternVL3/refcoco_id/stage1_multi_test \
        --output InternVL3/refcoco_id/identity_qa_pairs_multi.json
"""

import argparse

from InternVL3.refcoco_id.processor_stage2_multi import IdentityStage2MultiProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Multi: Refine and convert multi-person identity dataset"
    )
    parser.add_argument(
        "--stage1_multi_folder",
        type=str,
        required=True,
        help="Folder containing Stage 1 Multi JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="InternVL3/refcoco_id/dataset/identity_qa_pairs_multi.json",
        help="Output file for final conversation array",
    )

    args = parser.parse_args()

    # Initialize processor
    print("=" * 80)
    print("STAGE 2 MULTI: Refine and Convert Multi-Person Identity Dataset")
    print("=" * 80)
    print(f"Stage 1 Multi folder: {args.stage1_multi_folder}")
    print(f"Output file: {args.output}")
    print("=" * 80)

    processor = IdentityStage2MultiProcessor(
        stage1_multi_folder=args.stage1_multi_folder, output_file=args.output
    )

    # Process all files
    print("\nProcessing Stage 1 Multi outputs...")
    conversations = processor.process_all()

    # Save results
    print("\nSaving results...")
    processor.save_results(conversations)

    # Print metrics
    processor.print_metrics()

    # Print sample
    if len(conversations) > 0:
        print("\n" + "=" * 80)
        print("SAMPLE CONVERSATION (first entry)")
        print("=" * 80)
        import json

        print(json.dumps(conversations[0], indent=2, ensure_ascii=False))
        print("=" * 80)


if __name__ == "__main__":
    main()
