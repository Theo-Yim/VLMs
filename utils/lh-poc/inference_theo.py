"""
Compared to refcoco_main2.py, this version:
- processes images one-by-one and save single results immediately
- batch processing is applied (single image with multiple objects and questions in batch)
"""

import argparse
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
    Main execution function
    """
    parser = argparse.ArgumentParser(description="InternVL 3.5 38B Inference")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/nas1/data/lh-poc/",
        help="Path to label data root",
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
    parser.add_argument("--gpu_id", type=int, default=5, help="GPU ID to use for this process. -1 for distributing all GPUs")
    args = parser.parse_args()

    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)

    # processor = RefCOCOProcessor(model_path="OpenGVLab/InternVL3_5-38B")
    model, tokenizer = load_models(args.model_path, device_map=f"cuda:{args.gpu_id}" if args.gpu_id != -1 else None)
    if args.enable_thinking:
        model.system_message = R1_SYSTEM_PROMPT

    # Load pre-merged datasets
    # data_list = processor.load_datasets()
    print("Loading data...")
    loader = LHDataLoader(args.data_root, type="train")

    print(f"Total items to process: {len(loader)}")
    if args.limit:
        print(f"Limited to first {args.limit} items")

    processed_count = 0
    for idx, item in enumerate(tqdm(loader, desc="Processing images")):
        if args.limit and idx >= args.limit:
            break

        # Get metadata
        data_key = item["label_id"]
        # label_id = item["label_id"]

        # Check if result already exists
        result_path = os.path.join(args.result_dir, f"{data_key}.txt")
        if os.path.exists(result_path) and not args.rerun:
            continue

        # Get image path
        image_path = os.path.join(loader.image_path, data_key + ".jpg")
        pixels = load_image(image_path, max_num=12).to(torch.bfloat16).to(f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cuda:0")

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

        print(f"\nProcessing {idx + 1}/{len(loader)}: {data_key}")
        print(f"Label ID: {data_key}")
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

        # if bbox is present, inference one more time with the bbox
        # Load and preprocess image
        if "annotation" in item["annotation_data"] and 'coord' in item["annotation_data"]["annotation"]:
            bbox = item["annotation_data"]["annotation"]["coord"]
            bbox = [bbox["x"], bbox["y"], bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]]
        else:
            continue
        pixels = load_image(image_path, max_num=12, bbox_xyxy=bbox).to(torch.bfloat16).to(f"cuda:{args.gpu_id}" if args.gpu_id != -1 else "cuda:0")
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt_1, generation_config)
        with open(result_path[:-4] + "_bbox.txt", "a", encoding="utf-8") as f:
            f.write(response + "\n" + str(bbox))

        processed_count += 1
        if processed_count >= args.limit:
            break

    print(f"\nProcessed {processed_count} images")


if __name__ == "__main__":
    main()
