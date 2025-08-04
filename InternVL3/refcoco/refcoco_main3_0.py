"""
Compared to refcoco_main2.py, this version:
- processes images one-by-one and save single results immediately
- batch processing is applied (single image with multiple objects and questions in batch)
"""

import json
import os

import torch
from tqdm import tqdm

from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_main import RefCOCOProcessor


def main():
    """
    Main execution function

    Key insight: RefCOCO datasets share the same COCO annotations (identical bboxes),
    but have different referring expressions. We merge these expressions rather than bboxes.

    Note: Run merge_refcoco_datasets.py first to create the merged dataset file.
    """
    processor = RefCOCOProcessor(model_path="OpenGVLab/InternVL3-38B")

    # Load pre-merged datasets
    data_list = processor.load_datasets()
    data_list = data_list[:int(len(data_list) * 0.1)]  # For testing, use only 10% of the dataset
    print(f"Loaded {len(data_list)} unique images with merged referring expressions")

    for data_entry in tqdm(data_list, desc="Processing images"):
        image_path = data_entry["image_path"]
        output_path = os.path.join(processor.output_folder, data_entry["image_id"] + ".json")
        if os.path.exists(output_path):
            print(f"Skipping {image_path}, already processed.")
            continue
        try:
            pixel_values = (
                load_image(os.path.join(processor.dataset_p_root, image_path), max_num=12)
                .to(torch.bfloat16)
                .cuda()
            )
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
        processor.generate_initial_questions_b(data_entry, pixel_values)
        processor.generate_detailed_responses_b(data_entry, pixel_values)
        # Save results
        # processor.save_results(data_entry, output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, indent=2, ensure_ascii=False)
        # print(f"Processed {image_path}.")


if __name__ == "__main__":
    main()
