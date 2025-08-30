"""
Compared to refcoco_main2.py, this version:
- processes images one-by-one and save single results immediately
- batch processing is applied (single image with multiple objects and questions in batch)
"""

import argparse
import os

import torch
from dataloader import LHDataLoader
from prompt import defect_types, material_parts, prompt_theo, spaces
from tqdm import tqdm

from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_utils import load_models


def extract_label_information(item):
    """
    Extract information dictionary from label data
    Same logic as mentioned by the user
    """
    information = {}
    if "categories" in item["label_data"] and "properties" in item["label_data"]["categories"]:
        properties = item["label_data"]["categories"]["properties"]
        for property in properties:
            if "property_name" in property:
                if "value" in property:
                    information[property["property_name"]] = property["value"]
                elif "option_names" in property:
                    information[property["property_name"]] = property["option_names"]

    return str(information)


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description="InternVL 3.5 38B Inference")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/nas1/data/lh-poc/lh-data/K-LH-302 2025-08-22 155843_export",
        help="Path to label data root",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/mnt/nas1/data/lh-poc/lh-data-image/image/20250722",
        help="Path to image root",
    )
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
    args = parser.parse_args()

    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)

    # processor = RefCOCOProcessor(model_path="OpenGVLab/InternVL3_5-38B")
    gpu_id = 0
    model, tokenizer = load_models(args.model_path, device_map=f"cuda:{gpu_id}")

    # Load pre-merged datasets
    # data_list = processor.load_datasets()
    print("Loading data...")
    loader = LHDataLoader(args.data_root, args.image_root)

    print(f"Total items to process: {len(loader)}")
    if args.limit:
        print(f"Limited to first {args.limit} items")

    processed_count = 0
    for idx, item in enumerate(tqdm(loader, desc="Processing images")):
        if args.limit and idx >= args.limit:
            break

        # Get metadata
        data_key = item["meta_data"]["data_key"]
        label_id = item["label_id"]

        # Check if result already exists
        result_path = os.path.join(args.result_dir, f"{label_id}.txt")
        if os.path.exists(result_path) and not args.rerun:
            continue

        # Get image path
        image_path = os.path.join(loader.image_root, data_key)
        pixels = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

        # Extract existing label information
        existing_labels = extract_label_information(item)

        prompt_1 = prompt_theo.format(
            spaces=spaces,
            material_parts=material_parts,
            defect_types=defect_types,
            existing_labels=existing_labels,
        )

        print(f"\nProcessing {idx + 1}/{len(loader)}: {data_key}")
        print(f"Label ID: {label_id}")
        print(f"Existing labels: {existing_labels}")
        generation_config = dict(
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, do_sample=True
        )
        # Run inference
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt_1, generation_config)
        # response = inference_engine.inference_single_image(
        #     image_path,
        #     "",  # Using the main prompt from prompt.py (referred to as prompt_theo)
        #     existing_labels
        # )

        # Save result
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(response)

        processed_count += 1
        if processed_count >= args.limit:
            break

    print(f"\nProcessed {processed_count} images")


if __name__ == "__main__":
    main()
