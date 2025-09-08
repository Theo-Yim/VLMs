import os
import json
import glob
from tqdm import tqdm
from dataloader import LHDataLoader
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
    
    # Correct incomplete <T tags to <T>
    # Find all occurrences of <T that are not followed by >
    content = re.sub(r'<T(?!>)', '<T>', content)
    content = re.sub(r'</T(?!>)', '</T>', content)
    content = re.sub(r'<A(?!>)', '<A>', content)
    content = re.sub(r'</A(?!>)', '</A>', content)

    answer = content.split("<A>")[-1]
    answer, bbox = answer.split("</A>")[0].strip(), answer.split("</A>")[1].strip()
    bbox = None if len(bbox) < 12 else bbox
    # answer = answer[answer.find("[")+1:answer.find("]")]
    answer = answer[answer.find("["):answer.rfind("]")+1]
    answer = answer.strip()
    answer = json.loads(answer)
    if len(answer) > 1:
        print(f" - Data Key: {img_name} has {len(answer)} defects. Using the first one only.")
        answer = answer[0]

    content = content[:content.find("<A>")].strip() + "\n" + json.dumps(answer, indent=1)

    content = content.replace("<T>", "<think>")
    content = content.replace("</T>", "</think>")
    content = content.replace("<A>", "<answer>")
    content = content.replace("</A>", "</answer>")
    
    return content, bbox


if __name__ == "__main__":
    data_root = "/mnt/nas1/data/lh-poc/"
    # image_root = "/mnt/nas1/data/lh-poc/lh-data-image/image/20250722"

    output_path = "/workspace/VLMs/utils/lh-poc/training_dataset.json"
    output_list = []

    # Load dataset
    loader = LHDataLoader(data_root)

    # Process only the assigned slice
    for item in tqdm(loader):

        # Get metadata
        index = item["index"]
        img_name = item["meta_data"]["data_key"]
        label_id = item["label_id"]
        # if 'objects' in item['label_data'] and len(item['label_data']['objects']) > 0:
        #     bbox = item['label_data']['objects'][0]['annotation']['coord']
        #     bbox = [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']]
        # else:
        #     bbox = [-1, -1, -1, -1]

        base_path = "/workspace/VLMs/utils/lh-poc/results_theo_parallel"
        found_files = find_label_files(base_path, label_id)
        if len(found_files) == 0:
            print(f"No found files for label_id: {label_id}")
            continue
        with open(found_files[0], "r") as f:
            content = f.read()
        content, _ = refine_content(content, img_name)

        # for file_path in found_files:
        #     print(f"Found: {file_path}")

        obj = {
            "id": f"sample_{index:05d}",
            "image": img_name,
            "bbox(xyxy)" : None,
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
        output_list.append(obj)
        
        if len(found_files) == 2:
            with open(found_files[1], "r") as f:
                content = f.read()
                content, bbox = refine_content(content, img_name)
                obj["bbox(xyxy)"] = bbox
                obj["conversations"][1]["value"] = content
                output_list.append(obj)

    with open(output_path, "a") as f:
        json.dump(output_list, f, indent=2, ensure_ascii=False)
        f.write("\n")

