import argparse
import json
import os

from spb_label import sdk
from spb_label.utils import SearchFilter, retrieve_file
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="train")
args = parser.parse_args()

rerun = True
type = args.type
# image_dir = f"/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-image-{type}/"
# annotation_dir = f"/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-annotation-{type}/"
image_dir = f"/workspace/data/lh-poc/lh-data-image-{type}/"
annotation_dir = f"/workspace/data/lh-poc/lh-data-annotation-{type}/"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)

source_tenant_id = "superbai-lh"
source_access_key = "0d48sngTavirJHirThOM1cJiRleNCrT1VaK48tUd"
source_project_name = "K-LH-302"

client = sdk.Client(
    project_name=source_project_name, team_name=source_tenant_id, access_key=source_access_key
)
filter = SearchFilter(project=client.project)
if type == "train":
    filter.tag_contains_all_of = ["f1ef3675-7a29-4c4b-a7ae-71b635d1a45e"]  # train_dataset
elif type == "test":
    filter.tag_contains_all_of = ["b5ad4965-ffdf-46a2-bdd3-1b4558d3cb68"]  # test_dataset
else:
    raise ValueError(f"Invalid type: {type}")
total_labels = client.get_num_labels(filter=filter)
pbar = tqdm(total=total_labels, desc="Fetching label handlers")


def download_labels(client, filter, cursor=None, pbar=None):
    count, labels, cursor = client.get_labels(filter=filter, cursor=cursor, page_size=10)
    for label in labels:
        # varies with other dataset
        id = str(label.get_id())
        object_labels = label.get_object_labels()
        tags = label.get_tags()
        if len(object_labels) == 0:
            annotation = {}
        else:
            annotation = object_labels[0]
        metadata = {}
        object = label.data.result["categories"]["properties"]
        for _object in object:
            if "option_names" in _object:
                metadata[_object["property_name"]] = _object["option_names"]
            if "value" in _object:
                metadata[_object["property_name"]] = _object["value"]
        annotation["metadata"] = metadata
        annotation["tags"] = tags
        image_url = label.get_image_url()
        image_path = os.path.join(image_dir, id + ".jpg")
        json_path = os.path.join(annotation_dir, id + ".json")
        if not os.path.exists(image_path):
            retrieve_file(url=image_url, file_path=image_path)
        if not os.path.exists(json_path) or rerun:
            with open(json_path, "w") as f:
                json.dump(annotation, f)
    if pbar is not None:
        pbar.update(len(labels))
    return cursor


cursor = download_labels(client, filter, cursor=None, pbar=pbar)
while cursor is not None:
    cursor = download_labels(client, filter, cursor=cursor, pbar=pbar)

# Verify downloaded data
print("\nVerifying downloaded files...")
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
json_files = [f for f in os.listdir(annotation_dir) if f.endswith(".json")]
print(f"Images: {len(image_files)}")
print(f"Annotations: {len(json_files)}")

# Check for broken files
broken_count = 0
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    if os.path.getsize(img_path) == 0:
        print(f"Warning: Empty image file {img_file}")
        broken_count += 1

for json_file in json_files:
    json_path = os.path.join(annotation_dir, json_file)
    if os.path.getsize(json_path) == 0:
        print(f"Warning: Empty JSON file {json_file}")
        broken_count += 1
    else:
        try:
            with open(json_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {json_file}: {e}")
            broken_count += 1

if broken_count == 0:
    print("✓ All files verified successfully!")
else:
    print(f"✗ Found {broken_count} broken files")
