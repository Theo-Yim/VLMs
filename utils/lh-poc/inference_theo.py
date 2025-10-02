"""
Compared to refcoco_main2.py, this version:
- processes images one-by-one and save single results immediately
- batch processing is applied (single image with multiple objects and questions in batch)
"""

import argparse
import gc
import os

import torch
from dataloader import LHDataLoader
from dataloader_pytorch_lh import create_dataloader
from name import DEFECT_CLASS, MATERIAL_CLASS, SPACE_CLASS
from prompt_sb_v2 import ENGLISH_TRAIN_PROMPT, R1_SYSTEM_PROMPT
from prompt_theo import defect_types, material_parts, prompt_theo_v2, prompt_theo_v2_system, spaces
from tqdm import tqdm

from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_utils import load_models


def extract_label_information(annotation_data):
    """
    Extract information dictionary from label data
    Same logic as mentioned by the user
    """

    is_no_defect = any(
        "하자유형_check" in tag or "NO(이미지 판단 불가)" in tag for tag in annotation_data["tags"]
    )
    properties = annotation_data["metadata"]  # categories.get("properties", [])
    information = {}
    if "공간" in properties and properties["공간"] in SPACE_CLASS:
        information["space"] = SPACE_CLASS[properties["공간"]]
    if "부위자재" in properties and properties["부위자재"] in MATERIAL_CLASS:
        information["material_parts"] = MATERIAL_CLASS[properties["부위자재"]]
    if "하자유형" in properties and properties["하자유형"] in DEFECT_CLASS:
        information["defect_types"] = DEFECT_CLASS[properties["하자유형"]]
    if "하자내용" in properties:
        information["defect_description"] = properties["하자내용"]

    if is_no_defect:
        information["defect_present"] = "Unknown"
        information["defect_type"] = "None"
        # information["defect_description"] = "None"
    else:
        information["defect_present"] = "Yes"
    return str(information)


def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description="InternVL 3.5 38B Inference")
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
        "--max_new_tokens", type=int, default=3072, help="Maximum number of new tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--rerun", action="store_true", help="Rerun inference for existing results")
    parser.add_argument(
        "--single_image", type=str, default=None, help="Path to single image for testing"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for this process. -1 for distributing all GPUs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    args = parser.parse_args()

    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)

    model, tokenizer = load_models(
        args.model_path, device_map=f"cuda:{args.gpu_id}" if args.gpu_id != -1 else None
    )
    if args.enable_thinking:
        model.system_message = prompt_theo_v2_system

    # Load dataset
    print("Loading data...")
    base_loader = LHDataLoader(args.data_root, type=args.data_type)

    # Create PyTorch DataLoader with multiprocessing workers
    device = f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cuda:0"
    loader = create_dataloader(base_loader, device=device, max_num=12, num_workers=args.num_workers)

    print(f"Total items to process: {len(base_loader)}")
    # Apply limit if specified
    if args.limit:
        print(f"Limited to first {args.limit} items")

    processed_count = 0
    error_count = 0
    for batch in tqdm(loader, desc="Processing images", total=len(base_loader)):
        item = batch[0]  # Batch size is 1
        idx = item.get("index", 0)  # Get index from item

        # Get metadata
        data_key = item["label_id"]
        # label_id = item["label_id"]

        # Check if result already exists
        result_path = os.path.join(args.result_dir, f"{data_key}.txt")
        if os.path.exists(result_path) and not args.rerun:
            continue

        # Check for loading errors
        if item.get("error"):
            print(f"Warning - Error loading image: {item['error']}")
            error_count += 1
            continue

        # Get prefetched pixels and move to GPU
        pixels = item["pixels"].to(device)

        # Extract existing label information
        existing_labels = extract_label_information(item["annotation_data"])

        prompt_sb = False
        if not prompt_sb:
            prompt_1 = prompt_theo_v2.format(
                spaces=spaces,
                material_parts=material_parts,
                defect_types=defect_types,
                existing_labels=existing_labels,
            )
        else:
            model.system_message = R1_SYSTEM_PROMPT
            prompt_1 = f"### Existing Label:\n{existing_labels}" + ENGLISH_TRAIN_PROMPT

        print(f"\nProcessing {idx + 1}/{len(loader)}: {data_key}")
        print(f"Label ID: {data_key}")
        print(f"Existing labels: {existing_labels}")
        generation_config = dict(
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, do_sample=True
        )
        # Run inference
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt_1, generation_config)

        # Save result
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(response)

        # if False:  # BBOX
        #     # if bbox is present, inference one more time with the bbox
        #     # Load and preprocess image
        #     if (
        #         "annotation" in item["annotation_data"]
        #         and "coord" in item["annotation_data"]["annotation"]
        #     ):
        #         bbox = item["annotation_data"]["annotation"]["coord"]
        #         bbox = [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
        #     else:
        #         continue
        #     pixels = (
        #         load_image(image_path, max_num=12, bbox_xyxy=bbox)
        #         .to(torch.bfloat16)
        #         .to(f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cuda:0")
        #     )
        #     with torch.inference_mode():
        #         response = model.chat(tokenizer, pixels, prompt_1, generation_config)
        #     with open(result_path[:-4] + "_bbox.txt", "a", encoding="utf-8") as f:
        #         f.write(response + "\n" + str(bbox))

        processed_count += 1
        if args.limit and processed_count >= args.limit:
            break

        # # Clean up GPU memory
        # del pixels
        # torch.cuda.empty_cache()
        # gc.collect()

    print(f"\nProcessed {processed_count} images")
    print(f"\nEncountered {error_count} errors")


if __name__ == "__main__":
    main()
