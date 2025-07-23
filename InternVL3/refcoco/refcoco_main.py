from datasets import load_dataset
from copy import deepcopy
from tqdm import tqdm
import torch
from PIL import Image
def xywh_to_xyxy(box):
    bbox = deepcopy(box)
    bbox[2] = box[0] + box[2]
    bbox[3] = box[1] + box[3]
    return bbox
# dataset = load_dataset("lmms-lab/RefCOCO")
data_path_lst = [
    "jxu124/RefCOCO",
    "jxu124/RefCOCOplus",
    "jxu124/RefCOCOg",
]

# data_path_lst = [
#     "lmms-lab/RefCOCO",
#     "lmms-lab/RefCOCOplus",
#     "lmms-lab/RefCOCOg",
# ]
for data_path in data_path_lst:
    print(f"Loading dataset: {data_path}")
    dataset = load_dataset(data_path)

    split_lst = dataset.keys()


    for split in split_lst:
        for sample in tqdm(dataset[split], desc=f"[{data_path}/{split}]"):
            image_path = sample["image_path"]
            image = Image.open(image_path).convert("RGB")

            texts = sample["text"]
            gt_bbox = xywh_to_xyxy(sample["bbox"])
            t_gt_bbox = torch.tensor(gt_bbox).view(-1, 4).to(dtype=torch.float32)
            assert len(t_gt_bbox) == 1, "sample['bbox'] should contain only a single box."

            for text in sample["answer"]:
                print(text)

