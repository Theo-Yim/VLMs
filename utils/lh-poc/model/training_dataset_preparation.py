import argparse
import glob
import json
import os
import re

from dataloader import LHDataLoader
from prompt_theo import defect_types, material_parts, prompt_inference, spaces
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

    # Correct incomplete <T tags to <T>
    # Find all occurrences of <T that are not followed by >
    content = re.sub(r"<T(?!>)", "<T>", content)
    content = re.sub(r"</T(?!>)", "</T>", content)
    content = re.sub(r"<A(?!>)", "<A>", content)
    content = re.sub(r"</A(?!>)", "</A>", content)

    answer = content.split("<A>")[-1]
    answer, bbox = answer.split("</A>")[0].strip(), answer.split("</A>")[1].strip()
    bbox = None if len(bbox) < 12 else bbox
    # answer = answer[answer.find("[")+1:answer.find("]")]
    answer = answer[answer.find("{") : answer.rfind("}") + 1]
    answer = answer.strip()
    answer = json.loads(answer)
    assert len(answer) < 8 and len(answer) > 1, (
        f" - Data Key: {img_name} has {len(answer)} defects. {answer}"
    )
    # if len(answer) > 1:
    #     print(f" - Data Key: {img_name} has {len(answer)} defects. Using the first one only.")
    #     answer = answer[0]

    content = content[: content.find("<A>")].strip() + "\n" + json.dumps(answer, indent=1)

    content = content.replace("<T>", "<think>")
    content = content.replace("</T>", "</think>")
    content = content.replace("<A>", "<answer>")
    content = content.replace("</A>", "</answer>")

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
        default="/workspace/VLMs/utils/lh-poc/results_theo_parallel_v2",
        help="Path to inference results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/VLMs/utils/lh-poc/dataset_lh_theo_train.json",
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
        default=0,
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
            try:
                content, _ = refine_content(raw_content, img_name)
            except Exception as e:
                if attempt == 0:
                    print(f"{result_file}, Error: {e}")
                content = None

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
                prompt_sb=False,
            )

            if not success:
                print(f"Inference failed for {label_id}")
                break

        if content is None:
            print(f"Skipping {label_id} after {max_retries} attempts")
            failed_items.append(label_id)
            continue

        user_message = f"<image>\n{prompt_inference.format(defect_message=defect_message, spaces=spaces, defect_types=defect_types, material_parts=material_parts)}"
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
