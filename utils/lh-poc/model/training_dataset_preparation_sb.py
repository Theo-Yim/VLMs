import argparse
import glob
import json
import os
import re

from dataloader import LHDataLoader
from prompt_sb_v2 import ENGLISH_INFERENCE_PROMPT
from prompt_theo import defect_types, material_parts, prompt_inference_R2, spaces
from rerun_failed_inference import rerun_single_item_inference
from tqdm import tqdm

#   {
#     "id": "sample_001",
#     "image": "sample_small.png",
#     "conversations": [
#       {
#         "from": "human",
#         "value": "<image>\nWhat do you see in this image?"
#       },
#       {
#         "from": "gpt",
#         "value": "I can see a mathematical diagram with various equations and figures. This appears to be an educational or instructional material related to mathematics."
#       }
#     ]
#   },


def find_label_files(base_path, label_id):
    """Find all files matching f'{label_id}.txt' in subfolders"""
    # pattern = os.path.join(base_path, "**", f"{label_id}.txt")
    # files = glob.glob(pattern, recursive=True)
    patterns = [
        os.path.join(base_path, "**", f"{label_id}.txt"),
        os.path.join(base_path, "**", f"{label_id}_bbox.txt"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files


def refine_content(content, img_name):
    """Refine the content"""

    content = content.replace("defect_present: No", "defect_present: Unknown").strip(" \n-")

    while content.endswith("</think>") or content.endswith("<think>"):
        if content.endswith("</think>"):
            content = content[: -len("</think>")].strip()
        else:
            content = content[: -len("<think>")].strip()

    think = content[: content.find("</think>") + len("</think>")].strip()
    if len(think) <= 20:
        if "Final answer:" in content:
            think = content[: content.find("Final answer: ")].strip() + "</think>"
            content = think + content[content.find("Final answer: ") :]
        else:
            print(f" - Data Key: {img_name} has invalid think: {think}")
            # print(f"   - Data Key: {img_name} does not have Final answer either. Skip this image.")
            return None, None
        # return None, None
    if len(think) > 3000:
        print(f" - Data Key: {img_name} has invalid think of too long.")
        return None, None

    # ================================
    # if there is any sentence including "existing labels" in the think sentences, remove the sentence.
    # First, check sentence by sentence in think sentences. Then if any sentence includes "existing labels", remove the sentence. Final think sentences should not include "existing labels".

    # Remove sentences containing "existing labels" from think content
    def remove_sentences_with_existing_labels(text):
        """Remove sentences containing 'existing labels' from text"""
        # Split into sentences considering various ending patterns
        sentences = re.split(r'[.!?]+["\']?\s*', text)
        # Filter out sentences containing "existing labels" (case insensitive)
        filtered_sentences = [
            s.strip() for s in sentences if s.strip() and "existing labels" not in s.lower()
        ]
        # Join back with periods
        return ". ".join(filtered_sentences)

    # Apply the filtering to think content
    think = remove_sentences_with_existing_labels(think)
    # ================================

    answer = content.split("</think>")[-1]
    if answer.strip().split("\n")[-1].strip().startswith("[") and answer.strip().split("\n")[
        -1
    ].strip().endswith("]"):
        # if the last line is a list, remove the last line. The last line is bbox.
        bbox = answer.strip().split("\n")[-1].strip()
        answer = "\n".join(answer.strip().split("\n")[:-1])
    else:
        answer = answer.strip()
        bbox = None
    answer = answer[answer.find("{") : answer.find("}") + 1]
    try:
        answer = json.loads(answer)
    except json.decoder.JSONDecodeError:
        print(f" - Data Key: {img_name} has invalid answer: {answer}")
        return None, None

    assert (len(answer) == 9 and answer["defect_present"] == "Yes") or (
        len(answer) >= 2
        and len(answer) <= 9
        and (answer["defect_present"] == "Unknown" or answer["defect_present"] == "No")
    ), f" - Data Key: {img_name} has {len(answer)} defects. {answer}"
    # if len(answer) != 7 and len(answer) != 3 and len(answer) != 2:
    #     print(f" - Data Key: {img_name} has {len(answer)} defects. {answer}")

    if bbox and len(bbox) < 12:
        # alert the user
        print(f" - Data Key: {img_name} has invalid bbox: {bbox}")
        bbox = None if len(bbox) < 12 else bbox

    content = think + "\n" + json.dumps(answer, ensure_ascii=False, indent=1)

    return content, bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset from inference results")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/Theo-Yim/data/lh-poc/",
        help="Path to label data root",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/workspace/data/lh-poc/results_dataset_creation/results_sb_parallel_v2",
        help="Path to inference results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/data/lh-poc/dataset_train_ready/dataset_lh_sb_v2_train.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="OpenGVLab/InternVL3_5-38B",
        help="Model path for auto rerun",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="GPU ID for auto rerun",
    )
    args = parser.parse_args()

    data_root = args.data_root
    base_path = args.base_path
    output_path = args.output_path
    output_list = []
    failed_items = []

    # Load dataset
    loader = LHDataLoader(data_root, "train")

    # Process only the assigned slice
    for item in tqdm(loader):
        # Get metadata
        index = item["index"]
        # img_name = item["meta_data"]["data_key"]
        label_id = item["label_id"]
        defect_message = item["annotation_data"]["metadata"]["하자내용"]

        img_name = label_id + ".jpg"

        # if label_id != "72672750-ee38-41d6-8412-4ea341795346":
        #     continue
        # if image_name not exists in image_root, skip
        image_path = os.path.join(data_root, "lh-data-image-train", img_name)
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            continue

        # Find result file once
        found_files = find_label_files(base_path, label_id)
        if len(found_files) == 0:
            print(f"No found files for label_id: {label_id}")
            continue

        result_file = found_files[0]

        # Try to get valid content, rerun inference if needed
        content = None
        max_retries = 3
        for attempt in range(max_retries):
            # Read and refine content
            with open(result_file, "r") as f:
                raw_content = f.read()
            content, _ = refine_content(raw_content, img_name)

            if content is not None:
                break  # Success!

            # Content is None, rerun inference (overwrites result_file)
            print(
                f"Refine failed for {label_id} (attempt {attempt + 1}/{max_retries}), rerunning inference..."
            )

            success = rerun_single_item_inference(
                label_id=label_id,
                item=item,
                result_path=result_file,
                model_path=args.model_path,
                gpu_id=args.gpu_id,
                prompt_sb=True,
            )

            if not success:
                print(f"Inference failed for {label_id}")
                break

        if content is None:
            print(f"Skipping {label_id} after {max_retries} attempts")
            failed_items.append(label_id)
            continue

        user_message = f"<image>\n{ENGLISH_INFERENCE_PROMPT.format(defect_message=defect_message, spaces=spaces, defect_types=defect_types, material_parts=material_parts)}"
        obj = {
            "id": f"sample_{index:05d}",
            "image": img_name,
            "conversations": [
                {
                    "from": "human",
                    "value": user_message,
                },
                {"from": "gpt", "value": content},
            ],
        }
        output_list.append(obj)
        # for file_path in found_files:
        #     print(f"Found: {file_path}")
        # output_list.append(copy.deepcopy(obj)) # append the copy of the obj
        # if len(found_files) == 2:
        #     with open(found_files[1], "r") as f:
        #         content = f.read()
        #         content, bbox = refine_content(content, img_name)
        #         if content is None:
        #             continue
        #         obj["bbox(xyxy)"] = bbox
        #         obj["conversations"][1]["value"] = content
        #         output_list.append(obj)

    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Save failed items to file if any
    if failed_items:
        failed_items_path = output_path.replace(".json", "_failed.txt")
        with open(failed_items_path, "w") as f:
            for label_id in failed_items:
                f.write(f"{label_id}\n")
        print(f"\nProcessed {len(output_list)} successful items")
        print(f"Found {len(failed_items)} permanently failed items, saved to {failed_items_path}")
    else:
        print(f"\nProcessed {len(output_list)} successful items")
        print("All items processed successfully!")
