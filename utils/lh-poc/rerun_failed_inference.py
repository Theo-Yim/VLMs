"""
Rerun inference for failed items from training dataset preparation.
"""

import os

import torch
from name import DEFECT_CLASS, MATERIAL_CLASS, SPACE_CLASS
from prompt_sb_v2 import ENGLISH_TRAIN_PROMPT, R1_SYSTEM_PROMPT
from prompt_theo import defect_types, material_parts, prompt_theo_v2, prompt_theo_v2_system, spaces

from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_utils import load_models

# Module-level cache for model and tokenizer
_model_cache = {}


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


def rerun_single_item_inference(
    label_id,
    item,
    result_path,
    model_path="OpenGVLab/InternVL3_5-38B",
    gpu_id=0,
    max_new_tokens=3072,
    temperature=0.6,
    prompt_sb=False,
):
    """
    Rerun inference for a single item and save to result_path

    Args:
        label_id: Label ID to reprocess
        item: Data item from loader
        result_path: Full path to save result
        model_path: Model path (cached after first load)
        gpu_id: GPU ID to use
        max_new_tokens: Maximum number of new tokens
        temperature: Sampling temperature
        prompt_sb: Whether to use prompt_sb

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load model once and cache it
        cache_key = (model_path, gpu_id)
        if cache_key not in _model_cache:
            print(f"Loading model {model_path} on GPU {gpu_id}...")
            model, tokenizer = load_models(
                model_path, device_map=f"cuda:{gpu_id}" if gpu_id != -1 else None
            )
            model.system_message = prompt_theo_v2_system
            device = f"cuda:{gpu_id}" if gpu_id != -1 else "cuda:0"
            _model_cache[cache_key] = (model, tokenizer, device)
        else:
            model, tokenizer, device = _model_cache[cache_key]

        # Check for loading errors
        if item.get("error"):
            print(f"Warning - Error loading image: {item['error']}")
            return False

        # Get image path from item (from LHDataLoader)
        image_path = item.get("image_file")
        if not image_path or not os.path.exists(image_path):
            print(f"Warning - Image file not found: {image_path}")
            return False
        
        # Load and preprocess image (same as inference_theo.py would do via dataloader)
        pixels = load_image(image_path, max_num=12).to(torch.bfloat16).to(device)

        # Extract existing label information
        existing_labels = extract_label_information(item["annotation_data"])

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

        generation_config = dict(
            max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True
        )

        # Run inference
        with torch.inference_mode():
            response = model.chat(tokenizer, pixels, prompt_1, generation_config)

        # Save result (overwrites the file)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(response)

        return True
    except Exception as e:
        print(f"Error during inference for {label_id}: {e}")
        return False
