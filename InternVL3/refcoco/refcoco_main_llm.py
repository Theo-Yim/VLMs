"""
Compared to refcoco_main2.py, this version:
- processes images one-by-one and save single results immediately
- batch processing is applied (single image with multiple objects and questions in batch)
"""

import gc
import json
import os

import torch
from tqdm import tqdm

from InternVL3.utils.processor_main import RefCOCOProcessor

# def fix_a2_string(data_entry):
#     """Fix A2 string and store the result in A3"""
#     anno = data_entry["annos_str"]
#     for qna in data_entry["QnA"]:
#         if "A3" in qna and len(qna["A3"]) > 100:
#             continue
#         qna["A3"] = fix_tool_calling_strings(data_entry["image_id"], qna["A2"], anno)
#     return


def main():
    """
    Main execution function

    Key insight: RefCOCO datasets share the same COCO annotations (identical bboxes),
    but have different referring expressions. We merge these expressions rather than bboxes.

    Note: Run merge_refcoco_datasets.py first to create the merged dataset file.
    """
    processor = RefCOCOProcessor(
        model_path="Qwen/Qwen3-14B-FP8",
    )  # Qwen3-30B-A3B-Thinking-2507-FP8")
    # output_folder = "/mnt/nas1/data/coco/refcoco_vlm_results_theo"
    output_dir = processor.output_folder.rstrip("/") + "_llm"

    # Find all JSON files in the output directory
    json_files = []
    for root, dirs, files in os.walk(processor.output_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    print(f"Loaded {len(json_files)} unique images with merged referring expressions")

    for json_path in json_files:  # tqdm(json_files, desc="Processing images"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data_entry = json.load(f)
            # Validate required fields
            required_fields = ["image_path", "image_id", "annos_str", "QnA"]
            for field in required_fields:
                if field not in data_entry:
                    print(f"Warning: Missing required field '{field}' in {json_path}")
                    continue
        except Exception as e:
            print(f"Error loading {json_path}: {e}")

        output_path = os.path.join(output_dir, data_entry["image_id"] + ".json")
        # fix_a2_string(data_entry)
        processor.fix_answer_strings(data_entry)
        torch.cuda.empty_cache()
        gc.collect()
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, indent=2, ensure_ascii=False)
        # print(f"Processed {image_path}.")


if __name__ == "__main__":
    main()
