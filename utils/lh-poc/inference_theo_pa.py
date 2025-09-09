"""
Compared to refcoco_main2.py, this version:
- processes images one-by-one and save single results immediately
- batch processing is applied (single image with multiple objects and questions in batch)
- ADDED: Support for parallel processing with multiple GPUs
"""

import argparse
import gc
import json
import os

import torch
from dataloader import LHDataLoader
from prompt_theo import defect_types, material_parts, prompt_theo, spaces, R1_SYSTEM_PROMPT
from prompt_sb import ENGLISH_TRAIN_PROMPT
from tqdm import tqdm

from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_utils import load_models


def extract_label_information(annotation_data):
    """
    Extract information dictionary from label data
    Same logic as mentioned by the user
    """
    information = {}
    target_dict = {
        "공간": "space",
        "부위자재": "material_parts",
        "하자유형": "defect_types",
        "하자내용": "defect_description",
    }

    # Optimized: Use any() with generator expression for early exit
    is_no_defect = any("NO(이미지 판단 불가)" in tag for tag in annotation_data["tags"])

    # Optimized: Cache label_data access and use more efficient property processing
    # label_data = annotation_data.get("label_data", {})
    # categories = label_data.get("categories", {})
    properties =  annotation_data['metadata']  # categories.get("properties", [])

    # # Optimized: Use dict comprehension for better performance
    # property_mapping = {
    #     prop["property_name"]: prop.get("value") or prop.get("option_names")
    #     for prop in properties
    #     if "property_name" in prop and prop["property_name"] in target_dict
    # }
    # Map properties to information dict
    for korean_name, english_name in target_dict.items():
        if korean_name in properties:
            information[english_name] = properties[korean_name]

    if is_no_defect:
        information["defect_present"] = "Unknown"
        information["defect_type"] = "None"
        information["defect_description"] = "None"
    else:
        information["defect_present"] = "Yes"
    return str(information)


def main():
    """
    Main execution function with parallel processing support
    """
    parser = argparse.ArgumentParser(description="InternVL 3.5 38B Parallel Inference")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/nas1/data/lh-poc/",
        help="Path to label data root",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="train",
        help="Type of data to process",
    )
    # parser.add_argument(
    #     "--image_root",
    #     type=str,
    #     default="/mnt/nas1/data/lh-poc/lh-data-image/image/20250722",
    #     help="Path to image root",
    # )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/workspace/VLMs/utils/lh-poc/results_test",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model_path", type=str, default="OpenGVLab/InternVL3_5-38B", help="HuggingFace model path"
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", help="Enable thinking mode with R1 system prompt"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--rerun", action="store_true", help="Rerun inference for existing results")
    parser.add_argument(
        "--single_image", type=str, default=None, help="Path to single image for testing"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")

    # NEW: Parallel processing arguments
    parser.add_argument("--gpu_id", type=int, default=5, help="GPU ID to use for this process")
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Starting index in dataset for this process"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="Ending index in dataset for this process"
    )
    parser.add_argument(
        "--process_id", type=int, default=0, help="Process ID for logging and unique identification"
    )

    args = parser.parse_args()

    # Set GPU visibility for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Process {args.process_id}: Using GPU {args.gpu_id}")

    # Create result directory with process-specific subdirectory
    process_result_dir = os.path.join(args.result_dir, f"process_{args.process_id}")
    os.makedirs(process_result_dir, exist_ok=True)
    print(f"Process {args.process_id}: Results will be saved to {process_result_dir}")

    # Load model on the assigned GPU (now GPU 0 from CUDA_VISIBLE_DEVICES perspective)
    device_map = f"cuda:0"  # Always 0 since we set CUDA_VISIBLE_DEVICES
    model, tokenizer = load_models(args.model_path, device_map=device_map)
    print(f"Process {args.process_id}: Model loaded successfully")
    if args.enable_thinking:
        model.system_message = R1_SYSTEM_PROMPT

    # Load dataset
    print(f"Process {args.process_id}: Loading data...")
    loader = LHDataLoader(args.data_root, type=args.data_type)
    total_items = len(loader)

    # Calculate data slice for this process
    if args.end_idx is None:
        args.end_idx = total_items

    # Validate indices
    args.start_idx = max(0, args.start_idx)
    args.end_idx = min(total_items, args.end_idx)

    if args.start_idx >= args.end_idx:
        print(
            f"Process {args.process_id}: No data to process (start_idx={args.start_idx}, end_idx={args.end_idx})"
        )
        return

    process_items = args.end_idx - args.start_idx
    print(
        f"Process {args.process_id}: Processing items {args.start_idx} to {args.end_idx - 1} ({process_items} items)"
    )

    # Apply limit if specified
    if args.limit and args.limit < process_items:
        args.end_idx = args.start_idx + args.limit
        process_items = args.limit
        print(f"Process {args.process_id}: Limited to {process_items} items")

    processed_count = 0
    error_count = 0

    # Process only the assigned slice
    for idx in tqdm(
        range(args.start_idx, args.end_idx),
        desc=f"Process {args.process_id}",
        position=args.process_id,
    ):
        item = loader[idx]

        # Get metadata
        data_key = item["label_id"]
        # label_id = item["label_id"]

        # Check if result already exists
        result_path = os.path.join(process_result_dir, f"{data_key}.txt")
        if os.path.exists(result_path) and not args.rerun:
            continue

        # Get image path
        image_path = os.path.join(loader.image_path, data_key + ".jpg")
        if not os.path.exists(image_path):
            print(f"Process {args.process_id}: Warning - Image not found: {image_path}")
            error_count += 1
            continue

        pixels = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

        # Extract existing label information
        existing_labels = extract_label_information(item["annotation_data"])

        prompt_sb = True
        if not prompt_sb:
            prompt_1 = prompt_theo.format(
                spaces=spaces,
                material_parts=material_parts,
                defect_types=defect_types,
                existing_labels=existing_labels,
            )
        else:
            prompt_1 = f"### Existing Label:\n{existing_labels}" + ENGLISH_TRAIN_PROMPT

        if (idx - args.start_idx) % 10 == 0:  # Print every 10th item
            print(f"Process {args.process_id}: Processing {idx + 1}/{total_items}: {data_key}")
            print(f"Process {args.process_id}: Label ID: {data_key}")

        generation_config = dict(
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, do_sample=True
        )
        # Run inference
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt_1, generation_config)

        # Save result
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(response)

        # if bbox is present, inference one more time with the bbox
        # Load and preprocess image
        if "annotation" in item["annotation_data"] and 'coord' in item["annotation_data"]["annotation"]:
            bbox = item["annotation_data"]["annotation"]["coord"]
            bbox = [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
        else:
            continue
        pixels = load_image(image_path, max_num=12, bbox_xyxy=bbox).to(torch.bfloat16).cuda()
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt_1, generation_config)
        with open(result_path[:-4] + "_bbox.txt", "a", encoding="utf-8") as f:
            f.write(response + "\n" + str(bbox))

        processed_count += 1

        # Clean up GPU memory
        del pixels
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Process {args.process_id}: Completed!")
    print(f"Process {args.process_id}: Processed {processed_count} images successfully")
    print(f"Process {args.process_id}: Encountered {error_count} errors")

    # Save process summary
    summary_path = os.path.join(process_result_dir, f"process_{args.process_id}_summary.json")
    summary = {
        "process_id": args.process_id,
        "gpu_id": args.gpu_id,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "total_assigned": process_items,
        "processed_successfully": processed_count,
        "errors": error_count,
        "result_dir": process_result_dir,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
