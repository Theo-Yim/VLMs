"""
RefCOCO Dataset Merging Script

This script loads RefCOCO, RefCOCOplus, and RefCOCOg datasets and merges their
referring expressions for identical COCO annotations. The merged data is saved
to be loaded by the main processing script.

Run this script once to create the merged dataset file.
"""

import pickle
import re
from collections import defaultdict
from copy import deepcopy

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# COCO Categories mapping (80 categories)
COCO_CATEGORIES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}


def xywh_to_xyxy(box):
    bbox = deepcopy(box)
    bbox[2] = box[0] + box[2]
    bbox[3] = box[1] + box[3]
    return bbox


class DescriptionMerger:
    def __init__(self):
        """Initialize with sentence transformer for semantic similarity"""
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        except:
            print("Warning: Could not load sentence transformer. Using basic merging.")
            self.sentence_model = None

    def remove_redundant_descriptions(self, descriptions, similarity_threshold=0.85):
        """Remove semantically similar descriptions"""
        if not self.sentence_model or len(descriptions) <= 1:
            return list(set(descriptions))  # Basic deduplication

        # Get embeddings
        embeddings = self.sentence_model.encode(descriptions)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Keep track of which descriptions to keep
        to_keep = [True] * len(descriptions)

        for i in range(len(descriptions)):
            if not to_keep[i]:
                continue
            for j in range(i + 1, len(descriptions)):
                if similarity_matrix[i][j] > similarity_threshold:
                    # Keep the longer, more descriptive one
                    if len(descriptions[i]) >= len(descriptions[j]):
                        to_keep[j] = False
                    else:
                        to_keep[i] = False
                        break

        return [desc for i, desc in enumerate(descriptions) if to_keep[i]]

    def prioritize_descriptions(self, descriptions):
        """Prioritize more informative descriptions"""

        # Sort by length and informativeness
        def description_score(desc):
            score = len(desc)  # Longer descriptions often more informative

            # Bonus for spatial information
            spatial_words = ["left", "right", "front", "back", "center", "middle", "top", "bottom"]
            score += sum(5 for word in spatial_words if word in desc.lower())

            # Bonus for color information
            color_w = ["red", "blue", "green", "yellow", "black", "white", "brown", "gray", "pink"]
            score += sum(3 for word in color_w if word in desc.lower())

            # Bonus for action/state information
            action_words = ["sitting", "standing", "walking", "holding", "wearing", "looking"]
            score += sum(4 for word in action_words if word in desc.lower())

            return score

        return sorted(descriptions, key=description_score, reverse=True)

    def merge_descriptions(self, description_lists):
        """Merge multiple description lists intelligently"""
        # Flatten all descriptions
        all_descriptions = []
        for desc_list in description_lists:
            all_descriptions.extend(desc_list)

        # Remove exact duplicates first
        unique_descriptions = list(set(all_descriptions))

        # Remove semantically similar descriptions
        filtered_descriptions = self.remove_redundant_descriptions(unique_descriptions)

        # Prioritize remaining descriptions
        final_descriptions = self.prioritize_descriptions(filtered_descriptions)

        # Limit to top 5 most informative descriptions
        return final_descriptions[:5]


class RefCOCODatasetMerger:
    def __init__(self):
        self.description_merger = DescriptionMerger()

    def get_image_id_from_path(self, image_path):
        """Extract image ID from path for deduplication"""
        # Example: coco/train2017/000000549347.jpg -> 549347
        match = re.search(r"(\d+)\.jpg", image_path)
        return match.group(1) if match else image_path

    def load_and_merge_datasets(self):
        """Load and process RefCOCO datasets with intelligent merging"""
        data_path_lst = [
            ("jxu124/RefCOCO", "RefCOCO"),
            ("jxu124/RefCOCOplus", "RefCOCOplus"),
            ("jxu124/RefCOCOg", "RefCOCOg"),
        ]

        # First pass: collect all data grouped by image ID
        image_data = defaultdict(lambda: defaultdict(list))

        for data_path, dataset_name in data_path_lst:
            print(f"Loading dataset: {data_path}")
            dataset = load_dataset(data_path)

            for split in dataset.keys():
                for sample in tqdm(dataset[split], desc=f"Processing {dataset_name}/{split}"):
                    image_path = sample["image_path"]
                    image_path = image_path.replace(
                        "train2014/COCO_train2014_", "train2017/"
                    )  # Temporary code because of the local foldername issue.
                    image_id = self.get_image_id_from_path(image_path)

                    # Get category from category_id using COCO mapping
                    category_id = sample.get("category_id")
                    category = COCO_CATEGORIES.get(category_id, "object")

                    bbox = sample["bbox"]  # xywh_to_xyxy(sample["bbox"])
                    descriptions = (
                        sample["captions"]
                        if isinstance(sample["captions"], list)
                        else [sample["captions"]]
                    )

                    annotation = {
                        "category": category,
                        "category_id": category_id,
                        "bbox": bbox,
                        "descriptions": descriptions,
                        "dataset_source": dataset_name,
                        "image_path": image_path,
                    }

                    image_data[image_id]["annotations"].append(annotation)
                    # Keep track of the image path (use the first one encountered)
                    if "image_path" not in image_data[image_id]:
                        image_data[image_id]["image_path"] = image_path

        print(f"Found {len(image_data)} unique images across all datasets")

        # Second pass: merge identical annotations and create final structure
        final_data = []

        for image_id, data in tqdm(image_data.items(), desc="Merging identical annotations"):
            merged_annotations = self.merge_identical_annotations(data["annotations"])

            if merged_annotations:  # Only include images with valid annotations
                anno_string = self.create_anno_string(merged_annotations)

                data_entry = {
                    "image_path": data["image_path"],
                    "image_id": image_id,
                    "annos_str": anno_string,
                    "annotations": merged_annotations,
                    "QnA": [],
                }
                final_data.append(data_entry)

        print(
            f"Final dataset contains {len(final_data)} unique images with merged referring expressions"
        )
        return final_data

    def merge_identical_annotations(self, annotations, bbox_tolerance=5):
        """Merge annotations with identical/nearly identical bboxes (same COCO annotations)

        Since RefCOCO datasets use the same underlying COCO annotations, the bounding boxes
        should be identical or nearly identical. The value is in merging diverse referring expressions.
        """
        merged = []
        used = [False] * len(annotations)

        for i, ann1 in enumerate(annotations):
            if used[i]:
                continue

            # Find all annotations with identical bboxes and categories
            identical = [ann1]
            used[i] = True

            for j, ann2 in enumerate(annotations[i + 1 :], i + 1):
                if used[j] or ann1["category"] != ann2["category"]:
                    continue

                # Check if bboxes are nearly identical (allowing small tolerance for floating point differences)
                bbox_diff = [abs(ann1["bbox"][k] - ann2["bbox"][k]) for k in range(4)]
                if all(diff <= bbox_tolerance for diff in bbox_diff):
                    identical.append(ann2)
                    used[j] = True

            # Merge identical annotations (mainly merging descriptions)
            if len(identical) > 1:
                merged_ann = self.merge_annotation_group(identical)
            else:
                merged_ann = identical[0]

            merged.append(merged_ann)

        return merged

    def merge_annotation_group(self, annotations):
        """Merge a group of identical annotations (same COCO annotation, different referring expressions)"""
        # Use the first annotation as base (bboxes should be nearly identical)
        base_ann = annotations[0]

        # Collect all descriptions from different datasets
        all_description_lists = [ann["descriptions"] for ann in annotations]

        # Merge descriptions intelligently
        merged_descriptions = self.description_merger.merge_descriptions(all_description_lists)

        # Use the first bbox (they should be nearly identical across datasets)
        bbox = base_ann["bbox"]

        # Collect source datasets
        sources = list(set(ann["dataset_source"] for ann in annotations))

        return {
            "category": base_ann["category"],
            "category_id": base_ann["category_id"],
            "bbox": bbox,  # Keep original bbox since they should be identical
            "descriptions": merged_descriptions,
            "dataset_sources": sources,
            "merged_from": len(annotations),
        }

    def create_anno_string(self, annotations):
        """Create annotation string with improved descriptions"""
        anno_parts = []
        object_counter = defaultdict(int)

        for ann in annotations:
            category = ann["category"]
            object_counter[category] += 1
            obj_num = object_counter[category]

            bbox_str = f"[{ann['bbox'][0]}, {ann['bbox'][1]}, {ann['bbox'][2]}, {ann['bbox'][3]}]"

            # Use merged, high-quality descriptions
            descriptions = ", ".join([f'"{desc}"' for desc in ann["descriptions"]])
            connector = "who is" if category == "person" else "which is"

            # Add source info for quality tracking (optional)
            source_info = (
                f" (merged from {ann.get('merged_from', 1)} sources)"
                if ann.get("merged_from", 1) > 1
                else ""
            )

            anno_parts.append(f"- {category} {obj_num} {bbox_str}, {connector} {descriptions}")

        return "\n".join(anno_parts)

    def save_merged_data(self, data_list, output_file="merged_refcoco_data.pkl"):
        """Save merged data to pickle file"""
        print(f"Saving merged data to {output_file}...")

        # Calculate merge statistics
        total_merged = sum(
            1
            for entry in data_list
            for ann in entry["annotations"]
            if ann.get("merged_from", 1) > 1
        )
        total_annotations = sum(len(entry["annotations"]) for entry in data_list)

        merge_stats = {
            "total_images": len(data_list),
            "total_annotations": total_annotations,
            "annotations_with_merged_expressions": total_merged,
            "merge_rate": total_merged / total_annotations * 100 if total_annotations > 0 else 0,
        }

        print(f"Merge Statistics:")
        print(f"Total images: {merge_stats['total_images']}")
        print(f"Total objects: {merge_stats['total_annotations']}")
        print(
            f"Objects with merged referring expressions: {merge_stats['annotations_with_merged_expressions']}"
        )
        print(f"Merge rate: {merge_stats['merge_rate']:.1f}%")

        # Save data with statistics
        merged_data = {"data": data_list, "merge_statistics": merge_stats}

        with open(output_file, "wb") as f:
            pickle.dump(merged_data, f)

        print(f"Merged data saved to {output_file}")
        return output_file


def main():
    """Main function to merge RefCOCO datasets"""
    print("Starting RefCOCO dataset merging...")

    merger = RefCOCODatasetMerger()

    # Load and merge datasets
    merged_data = merger.load_and_merge_datasets()

    # Save merged data
    output_file = merger.save_merged_data(merged_data)

    print(f"\nMerging complete! Use {output_file} in your main processing script.")


if __name__ == "__main__":
    main()
