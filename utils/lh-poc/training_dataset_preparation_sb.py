import os
import json
import glob
from tqdm import tqdm
from dataloader import LHDataLoader
import copy
import re
from prompt_theo import prompt_inference

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
        os.path.join(base_path, "**", f"{label_id}_bbox.txt")
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return files

def refine_content(content, img_name):
    """Refine the content"""

    think = content[:content.find("</think>") + len("</think>")].strip()
    if len(think) <= 20:
        print(f" - Data Key: {img_name} has invalid think: {think}")
        return None, None
    if len(think) > 2600:
        print(f" - Data Key: {img_name} has invalid think of too long.")
        return None, None
    answer = content.split("</think>")[-1]
    if answer.strip().split("\n")[-1].strip().startswith("[") and answer.strip().split("\n")[-1].strip().endswith("]"):
        # if the last line is a list, remove the last line. The last line is bbox.
        bbox = answer.strip().split("\n")[-1].strip()
        answer = "\n".join(answer.strip().split("\n")[:-1])
    else:
        answer = answer.strip()
        bbox = None
    answer = answer[answer.find("{"):answer.find("}")+1]
    try:
        answer = json.loads(answer)
    except json.decoder.JSONDecodeError:
        print(f" - Data Key: {img_name} has invalid answer: {answer}")
        return None, None
    if len(answer) != 7 and len(answer) != 3:
        print(f" - Data Key: {img_name} has {len(answer)} defects. {answer}")

    if bbox and len(bbox) < 12:
        # alert the user
        print(f" - Data Key: {img_name} has invalid bbox: {bbox}")
        bbox = None if len(bbox) < 12 else bbox

    content = think + "\n" + json.dumps(answer, ensure_ascii=False, indent=1)
    
    return content, bbox


if __name__ == "__main__":
    data_root = "/home/Theo-Yim/data/lh-poc/" # "/mnt/nas1/data/lh-poc/"
    # image_root = "/mnt/nas1/data/lh-poc/lh-data-image/image/20250722"

    output_path = "/workspace/VLMs/utils/lh-poc/dataset_lh_sb_train.json"
    output_list = []

    # Load dataset
    loader = LHDataLoader(data_root, "train")

    # Process only the assigned slice
    for item in tqdm(loader):

        # Get metadata
        index = item["index"]
        # img_name = item["meta_data"]["data_key"]
        label_id = item["label_id"]

        img_name = label_id + ".jpg"

        base_path = "/workspace/VLMs/utils/lh-poc/results_theo_parallel"
        found_files = find_label_files(base_path, label_id)
        if len(found_files) == 0:
            print(f"No found files for label_id: {label_id}")
            continue
        with open(found_files[0], "r") as f:
            content = f.read()
        content, _ = refine_content(content, img_name)

        if content is None:
            continue

        # for file_path in found_files:
        #     print(f"Found: {file_path}")

        obj = {
            "id": f"sample_{index:05d}",
            "image": img_name,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{prompt_inference}"
                },
                {
                    "from": "gpt",
                    "value": content
                }
            ]
        }
        output_list.append(copy.deepcopy(obj)) # append the copy of the obj
        
        if len(found_files) == 2:
            with open(found_files[1], "r") as f:
                content = f.read()
                content, bbox = refine_content(content, img_name)
                if content is None:
                    continue
                obj["bbox(xyxy)"] = bbox
                obj["conversations"][1]["value"] = content
                output_list.append(obj)

    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
        f.write("\n")

