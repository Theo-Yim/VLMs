import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# Global variable to cache COCO annotations to avoid reloading
_coco_annotations_cache = {}


def load_coco_annotations(image_id: str, coco_annotation_file: Optional[str] = None) -> Dict:
    """
    Load COCO annotations for a specific image_id.

    Args:
        image_id: The image ID (e.g., "000000581857")
        coco_annotation_file: Path to COCO annotation file (e.g., instances_train2017.json)
                              If None, will try to find common paths

    Returns:
        Dict containing COCO annotations for the image
    """
    global _coco_annotations_cache

    # Default COCO annotation file paths to try
    if coco_annotation_file is None:
        possible_paths = [
            "/mnt/nas-1/data/coco/annotations/instances_train2017.json",
            "/mnt/nas1/data/coco/annotations/instances_train2017.json",
            "/mnt/nas3/Data/coco/annotations/instances_train2017.json",
            "/workspace/VLMs/coco/annotations/instances_train2017.json",
            "coco/annotations/instances_train2017.json",
        ]

        coco_annotation_file = None
        for path in possible_paths:
            if os.path.exists(path):
                coco_annotation_file = path
                break

    # If no annotation file found, return empty dict
    if coco_annotation_file is None or not os.path.exists(coco_annotation_file):
        print(f"Warning: COCO annotation file not found. Checked paths: {possible_paths}")
        return {}

    # Load and cache COCO annotations if not already cached
    if coco_annotation_file not in _coco_annotations_cache:
        print(f"Loading COCO annotations from: {coco_annotation_file}")
        try:
            with open(coco_annotation_file, "r") as f:
                coco_data = json.load(f)

            # Create lookup by image_id
            image_annotations = {}
            for ann in coco_data.get("annotations", []):
                ann_ = {}
                ann_["id"] = ann["id"]
                ann_["category_id"] = ann["category_id"]
                ann_["bbox"] = ann["bbox"]
                ann_["area"] = ann["area"]
                ann_["iscrowd"] = ann["iscrowd"]
                ann_["image_id"] = ann["image_id"]
                img_id = ann["image_id"]
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann_)

            # Also create image info lookup
            images_info = {img["id"]: img for img in coco_data.get("images", [])}
            categories_info = {cat["id"]: cat for cat in coco_data.get("categories", [])}

            _coco_annotations_cache[coco_annotation_file] = {
                "annotations": image_annotations,
                "images": images_info,
                "categories": categories_info,
            }

        except Exception as e:
            print(f"Error loading COCO annotations: {e}")
            return {}

    # Get annotations for the specific image
    cached_data = _coco_annotations_cache[coco_annotation_file]

    # Convert string image_id to int for lookup
    try:
        image_id_int = int(image_id.lstrip("0"))  # Remove leading zeros
    except ValueError:
        print(f"Warning: Could not convert image_id '{image_id}' to int")
        return {}

    # Return annotations for this image
    annotations = cached_data["annotations"].get(image_id_int, [])
    image_info = cached_data["images"].get(image_id_int, {})
    categories = cached_data["categories"]

    ann_out = []
    for ann in annotations:
        ann_out.append(
            {
                "category_name": categories[ann["category_id"]]["name"],
                "bbox": ann["bbox"],
                "area": ann["area"],
            }
        )
    output = {
        "image_id": image_id,
        "image_height": image_info["height"],
        "image_width": image_info["width"],
        "annotations": ann_out,
    }
    # return {
    #     'image_id': image_id_int,
    #     'image_info': image_info,
    #     'annotations': annotations,
    #     'categories': categories
    # }
    return output


def parse_annotations(anno_str: str) -> Dict[str, List[str]]:
    """
    Parse annotation string to extract noun+number patterns and their descriptions.

    Returns:
        Dict mapping "noun number" to list of descriptions
    """
    annotations = {}

    # Split by lines and process each annotation
    lines = anno_str.strip().split("\n")

    for line in lines:
        if not line.strip() or not line.startswith("-"):
            continue

        # Extract noun phrase + number pattern (e.g., "person 1", "police car 2", "small dog 3")
        match = re.search(r"-\s*(.+?)\s+(\d+)\s*\[", line)
        if not match:
            continue

        noun_phrase = match.group(1).strip()
        number = match.group(2)
        noun_number = f"{noun_phrase} {number}"

        # Extract descriptions within quotes
        descriptions = re.findall(r'"([^"]+)"', line)

        annotations[noun_number] = descriptions

    return annotations


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def find_best_match(
    target_text: str, annotations: Dict[str, List[str]], threshold: float = 0.5
) -> Optional[str]:
    """
    Find the best matching noun+number for a given target text.
    Only returns a match if there's high confidence (threshold=0.5).

    Args:
        target_text: The text to match against descriptions
        annotations: Dict of noun+number to descriptions
        threshold: Minimum similarity threshold (conservative: 0.5)

    Returns:
        Best matching noun+number or None if no good match found
    """
    best_match = None
    best_score = threshold

    for noun_number, descriptions in annotations.items():
        # Edge case: if the target text is already a noun+number+... pattern, return it
        if noun_number in target_text:
            return noun_number

        for desc in descriptions:
            score = similarity_score(target_text, desc)
            if score > best_score:
                best_score = score
                best_match = noun_number

    # Additional conservative check: only return match if score is significantly high
    if best_match and best_score > threshold:
        return best_match

    return None


def extract_tool_calls(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract all tool-calling patterns from text.

    Returns:
        List of tuples: (tool_call_content, start_pos, end_pos)
    """
    tool_calls = []
    pattern = r"\{([^}]+)\}"

    for match in re.finditer(pattern, text):
        content = match.group(1).strip()
        tool_calls.append((content, match.start(), match.end()))

    return tool_calls


def is_valid_crop_format(content: str, valid_nouns: set) -> bool:
    """
    Check if tool-calling content follows correct "Crop noun number" format.
    """
    # Remove "Crop" prefix if present
    if content.lower().startswith("crop"):
        content = content[4:].strip()

    # when content is like "Crop person 1 [103.93, 299.99, 238.15, 477.41]"
    # remove the bbox part
    if content[-1] == "]":
        content = content[: content.rfind("[")].strip()
        # content = re.sub(r'\[[^\]]*\]', '', content)

    # Split by comma to handle multiple entities
    parts = [part.strip() for part in content.split(",")]

    for part in parts:
        # Check if it matches "noun phrase + number" pattern (e.g., "person 1", "tennis racket 1")
        match = re.match(r"^(.+?)\s+(\d+)$", part)
        if not match:
            return False

        noun_phrase = match.group(1).strip()
        number = match.group(2)
        noun_number = f"{noun_phrase} {number}"
        if noun_number not in valid_nouns:
            return False

    return True


def fix_tool_calling_content(content: str, annotations: Dict[str, List[str]]) -> str:
    """
    Fix a single tool-calling content string.
    Args:
        content: The tool-calling content to fix
        annotations: RefCOCO annotations dict
    """
    n_fixed = 0
    n_not_found = 0

    # Remove "Crop" prefix if present
    crop_prefix = ""
    if content.lower().startswith("crop"):
        crop_prefix = "Crop "
        content = content[4:].strip()

    # Split by comma to handle multiple entities
    parts = [part.strip() for part in content.split(",")]
    fixed_parts = []
    not_found_parts = []

    for part in parts:
        # Check if it already follows correct format
        match = re.match(r"^(\w+)\s+(\d+)$", part)
        if match:
            noun_number = f"{match.group(1)} {match.group(2)}"
            if noun_number in annotations:
                fixed_parts.append(part)
                n_fixed += 1
                continue

        # Try to find best match in annotations
        best_match = find_best_match(part, annotations)
        if best_match:
            fixed_parts.append(best_match)
            n_fixed += 1
        else:
            # If no match found, keep original but log warning
            # print(f"  Warning: Could not find match for '{part}'")
            n_not_found += 1
            fixed_parts.append(part)
            not_found_parts.append(part)

    fixed_parts_unique = list(dict.fromkeys(fixed_parts))  # Remove duplicates & order preserved
    return (
        crop_prefix + ", ".join(fixed_parts_unique),
        ", ".join(not_found_parts),
        n_fixed,
        n_not_found,
    )


def set_bbox(fixed_content: str, anno_str, annotations, coco_annotations: Dict = None):
    """Set bbox for the crop tool call.
    Args:
        fixed_content: The fixed string of object names and numbers
        anno_str: Merged RefCOCO annotation string
        annotations: RefCOCO annotations dict
        coco_annotations: Optional COCO annotations dict for additional context
                         Contains: {'image_id', 'image_height', 'image_width', 'annotations'}
                         coco_annotations['annotations'] contains: {'category_name', 'bbox', 'area'}
    Returns:
        bbox: [x1, y1, x2, y2] format
    """

    # corrected_anno_str = fix_double_conversion(anno_str, coco_annotations) # now it is called before fix_tool_calling_content
    corrected_anno_str = anno_str

    # Re-parse annotations after potential correction
    corrected_annotations = parse_annotations(corrected_anno_str)

    # Extract object names from fixed_content
    object_names = extract_object_names_from_content(fixed_content)

    if not object_names:
        print("  set_bbox: Warning: No object names found in fixed_content")
        return []

    # TODO2: Find bboxes for known and unknown objects
    all_bboxes = []

    # Find known objects (exist in annotations)
    known_objects = []

    # Use enhanced matching for both known and unknown objects
    matched_objects, unmatched_objects = find_matching_objects(
        object_names, corrected_annotations, corrected_anno_str, coco_annotations
    )

    # Add matched objects to all_bboxes
    for obj_name, matched_bboxes, match_type in matched_objects:
        all_bboxes.extend(matched_bboxes)
        for bbox in matched_bboxes:
            known_objects.append((obj_name, bbox))
    #  print(f"Found {len(matched_bboxes)} bbox(es) for '{obj_name}' (via {match_type})")

    # Handle remaining unmatched objects using proximity if we have known objects
    need_larger_area = False
    if (
        unmatched_objects
        and known_objects
        and coco_annotations
        and "annotations" in coco_annotations
    ):
        for unknown_obj in unmatched_objects:
            category_name = extract_category_name(unknown_obj)
            close_bboxes = find_close_coco_bbox(category_name, known_objects, coco_annotations)
            if close_bboxes:
                all_bboxes.extend(close_bboxes)
                print(
                    f"  set_bbox: Found {len(close_bboxes)} close COCO bbox for '{unknown_obj}' (via proximity)"
                )
            else:
                need_larger_area = True
            #     print(f"  set_bbox: No close COCO bbox found for '{unknown_obj}'. We will remove this crop entirely.")
            #     return []

    # Calculate encompassing bbox
    if all_bboxes:
        final_bbox = calculate_encompassing_bbox(all_bboxes)
        if need_larger_area:
            margin = int(
                coco_annotations["image_width"] * coco_annotations["image_height"] * 0.00015
            )
            final_bbox = [
                min(final_bbox[0] - margin, 0),
                min(final_bbox[1] - margin, 0),
                max(final_bbox[2] + margin, coco_annotations["image_width"]),
                max(final_bbox[3] + margin, coco_annotations["image_height"]),
            ]
        # print(f"Final encompassing bbox: {final_bbox}")

        obj_area = (final_bbox[2] - final_bbox[0]) * (final_bbox[3] - final_bbox[1])
        img_area = coco_annotations["image_width"] * coco_annotations["image_height"]
        if obj_area / img_area > 0.52:
            print(f"  set_bbox: Object area is too large ({obj_area / img_area:.2f})")
            final_bbox = []
        return final_bbox

    print("  Warning: No valid bboxes found")
    return []


def fix_double_conversion(anno_str: str, coco_annotations: Dict = None) -> str:
    """Fix double conversion issues where [x1,y1,x2,y2] should be [x1,y1,w,h]."""
    if not coco_annotations or "annotations" not in coco_annotations:
        return anno_str

    corrected_anno_str = anno_str

    # Extract all bboxes from anno_str
    bbox_pattern = r"\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]"
    matches = list(re.finditer(bbox_pattern, anno_str))

    for match in reversed(matches):  # Process in reverse to maintain string positions
        x1, y1, x2, y2 = map(float, match.groups())
        xyxy_bbox = [x1, y1, x2, y2]

        # Convert to xywh format to check against COCO
        xywh_bbox = [x1, y1, x2 - x1, y2 - y1]

        # Check if this matches any COCO annotation bbox
        for coco_ann in coco_annotations["annotations"]:
            coco_bbox = coco_ann["bbox"]  # Should be in [x, y, w, h] format
            coco_bbox = [
                coco_bbox[0],
                coco_bbox[1],
                coco_bbox[0] + coco_bbox[2],
                coco_bbox[1] + coco_bbox[3],
            ]

            # Allow small tolerance for floating point comparison
            if (
                abs(xywh_bbox[0] - coco_bbox[0]) < 1.0
                and abs(xywh_bbox[1] - coco_bbox[1]) < 1.0
                and abs(xywh_bbox[2] - coco_bbox[2]) < 1.0
                and abs(xywh_bbox[3] - coco_bbox[3]) < 1.0
            ):
                # Found a match - this bbox was likely double-converted
                # print(f"Found double-converted bbox: {xyxy_bbox} -> {xywh_bbox}")

                # Replace with xywh format
                new_bbox_str = f"[{xywh_bbox[0]:.2f}, {xywh_bbox[1]:.2f}, {xywh_bbox[2]:.2f}, {xywh_bbox[3]:.2f}]"
                corrected_anno_str = (
                    corrected_anno_str[: match.start()]
                    + new_bbox_str
                    + corrected_anno_str[match.end() :]
                )
                break

    return corrected_anno_str


def extract_object_names_from_content(fixed_content: str) -> List[str]:
    """Extract object names from fixed_content like 'Crop orange 2, apple 1'."""
    # Remove "Crop" prefix if present
    content = fixed_content
    if content.lower().startswith("crop"):
        content = content[4:].strip()

    # Split by comma and clean up
    object_names = [name.strip() for name in content.split(",")]
    return [name for name in object_names if name]


def extract_bbox_from_anno_str(obj_name: str, anno_str: str) -> Optional[List[float]]:
    """Extract bbox for a specific object from anno_str."""
    # Find the line containing this object
    lines = anno_str.strip().split("\n")

    for line in lines:
        if line.strip().startswith(f"- {obj_name} "):
            # Extract bbox from this line
            bbox_match = re.search(r"\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", line)
            if bbox_match:
                bbox = [float(x) for x in bbox_match.groups()]
                # Convert from [x, y, w, h] to [x1, y1, x2, y2] if needed
                if len(bbox) == 4:
                    # Assume it's in [x1, y1, x2, y2] format already from anno_str
                    return bbox

    return None


def extract_category_name(obj_name: str) -> str:
    """Extract category name from 'category number' pattern."""
    # Remove number and extra spaces
    parts = obj_name.split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return " ".join(parts[:-1])
    return obj_name


def find_matching_objects(
    object_names: List[str],
    corrected_annotations: Dict[str, List[str]],
    corrected_anno_str: str,
    coco_annotations: Dict = None,
) -> Tuple[List, List]:
    """Find matching objects using enhanced semantic matching.

    Handles:
    1. Exact matches in RefCOCO annotations
    2. Plural forms (e.g., "pizzas" → multiple "pizza" objects)
    3. Semantic similarity (e.g., "bookshelf" → "book", "plant" → "potted plant")

    Returns:
        matched_objects: List of (obj_name, bboxes, match_type) tuples
        unmatched_objects: List of unmatched object names
    """
    matched_objects = []
    unmatched_objects = []

    for obj_name in object_names:
        obj_matched = False

        # Strategy 1: Exact match in RefCOCO annotations
        if obj_name in corrected_annotations:
            bbox = extract_bbox_from_anno_str(obj_name, corrected_anno_str)
            if bbox:
                matched_objects.append((obj_name, [bbox], "exact_refcoco"))
                obj_matched = True
                continue

        # Strategy 2: Handle plural forms - find all objects of similar category
        plural_matches = find_plural_matches(obj_name, corrected_annotations, corrected_anno_str)
        if plural_matches:
            matched_objects.append((obj_name, plural_matches, "plural_refcoco"))
            obj_matched = True
            continue

        # Strategy 3: Semantic similarity in RefCOCO annotations
        semantic_matches = find_semantic_matches(
            obj_name, corrected_annotations, corrected_anno_str
        )
        if semantic_matches:
            matched_objects.append((obj_name, semantic_matches, "semantic_refcoco"))
            obj_matched = True
            continue

        # Strategy 4: Check COCO annotations for category matches
        if coco_annotations and "annotations" in coco_annotations:
            coco_matches = find_coco_category_matches(obj_name, coco_annotations)
            if coco_matches:
                matched_objects.append((obj_name, coco_matches, "coco_category"))
                obj_matched = True
                continue

        # If no matches found, add to unmatched
        if not obj_matched:
            unmatched_objects.append(obj_name)

    return matched_objects, unmatched_objects


def find_plural_matches(
    obj_name: str, annotations: Dict[str, List[str]], anno_str: str
) -> List[List[float]]:
    """Find all objects that match a plural form (e.g., 'pizzas' → all 'pizza' objects)."""
    # Convert plural to singular for common cases
    singular_candidates = get_singular_forms(obj_name)

    all_bboxes = []
    for singular in singular_candidates:
        # Find all annotation keys that start with this singular form
        for ann_key in annotations.keys():
            if ann_key.startswith(singular + " ") and ann_key.split()[0] == singular:
                bbox = extract_bbox_from_anno_str(ann_key, anno_str)
                if bbox:
                    all_bboxes.append(bbox)

    return all_bboxes


def find_semantic_matches(
    obj_name: str, annotations: Dict[str, List[str]], anno_str: str, threshold: float = 0.6
) -> List[List[float]]:
    """Find semantically similar objects using similarity scoring."""
    all_bboxes = []

    # Check each annotation key for semantic similarity
    for ann_key in annotations.keys():
        # Extract the category part (without number)
        category = extract_category_name(ann_key)

        # Check similarity between obj_name and category
        similarity = similarity_score(obj_name, category)
        if similarity > threshold:
            bbox = extract_bbox_from_anno_str(ann_key, anno_str)
            if bbox:
                all_bboxes.append(bbox)

        # Also check against descriptions
        for desc in annotations[ann_key]:
            similarity = similarity_score(obj_name, desc)
            if similarity > threshold:
                bbox = extract_bbox_from_anno_str(ann_key, anno_str)
                if bbox and bbox not in all_bboxes:
                    all_bboxes.append(bbox)
                break  # Don't add same bbox multiple times

    return all_bboxes


def find_coco_category_matches(
    obj_name: str, coco_annotations: Dict, threshold: float = 0.55
) -> List[List[float]]:
    """Find matching objects in COCO annotations based on category similarity."""
    all_bboxes = []

    # Special case mappings for common mismatches
    special_mappings = {
        "plant": ["potted plant"],
        "shelf": ["bookshelf"],
        "bookshelf": ["book"],
        "plants": ["potted plant"],
    }

    for coco_ann in coco_annotations.get("annotations", []):
        category_name = coco_ann.get("category_name", "")
        is_match = False

        # Check special mappings first
        if obj_name.lower() in special_mappings:
            if category_name in special_mappings[obj_name.lower()]:
                is_match = True

        # Check if category contains the object name or vice versa
        if not is_match:
            if (
                obj_name.lower() in category_name.lower()
                or category_name.lower() in obj_name.lower()
            ):
                is_match = True

        # Check direct similarity
        if not is_match:
            similarity = similarity_score(obj_name, category_name)
            if similarity > threshold:
                is_match = True

        if is_match:
            # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
            coco_bbox = coco_ann["bbox"]
            coco_xyxy = [
                coco_bbox[0],
                coco_bbox[1],
                coco_bbox[0] + coco_bbox[2],
                coco_bbox[1] + coco_bbox[3],
            ]
            all_bboxes.append(coco_xyxy)

    return all_bboxes


def get_singular_forms(word: str) -> List[str]:
    """Convert plural forms to singular. Handle common English plural rules."""
    singular_candidates = []

    # Handle common plural patterns
    if word.endswith("s"):
        # Remove 's' (books → book)
        singular_candidates.append(word[:-1])

        # Handle 'ies' → 'y' (candies → candy)
        if word.endswith("ies") and len(word) > 3:
            singular_candidates.append(word[:-3] + "y")

        # Handle 'es' → '' (boxes → box)
        if word.endswith("es") and len(word) > 2:
            singular_candidates.append(word[:-2])

    # Also include the original word in case it's already singular
    singular_candidates.append(word)

    return list(set(singular_candidates))  # Remove duplicates


def find_close_coco_bbox(
    category_name: str, known_objects: List[Tuple[str, List[float]]], coco_annotations: Dict
) -> List[List[float]]:
    """Find all close COCO bboxes for unknown object based on known objects.

    Uses improved proximity detection that checks:
    1. Bbox overlap (any overlap between rectangles)
    2. Side-to-side distance (minimum distance between bbox edges, not just centers)
    3. Returns ALL qualifying close bboxes, not just the closest one

    Args:
        category_name: Category to search for (e.g., 'apple', 'orange')
        known_objects: List of (name, bbox) tuples for known objects
        coco_annotations: COCO annotations dict with 'annotations' key

    Returns:
        List of close bboxes in [x1, y1, x2, y2] format
    """
    if not known_objects or "annotations" not in coco_annotations:
        return []

    close_bboxes = []
    proximity_threshold = 40  # 50  # pixels - reduced for tighter proximity

    # Get all known bboxes in xyxy format
    known_bboxes = [bbox for obj_name, bbox in known_objects]

    for coco_ann in coco_annotations["annotations"]:
        if coco_ann["category_name"] == category_name:
            # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
            coco_bbox = coco_ann["bbox"]
            coco_xyxy = [
                coco_bbox[0],
                coco_bbox[1],
                coco_bbox[0] + coco_bbox[2],
                coco_bbox[1] + coco_bbox[3],
            ]

            # Check if this COCO bbox is close to any known bbox
            is_close = False
            for known_bbox in known_bboxes:
                if bbox_overlaps(coco_xyxy, known_bbox):
                    print(f"  Found overlapping COCO bbox for '{category_name}': {coco_xyxy}")
                    is_close = True
                    break
                elif bbox_min_distance(coco_xyxy, known_bbox) <= proximity_threshold:
                    distance = bbox_min_distance(coco_xyxy, known_bbox)
                    print(
                        f"  Found close COCO bbox for '{category_name}': {coco_xyxy} (distance: {distance:.1f})"
                    )
                    is_close = True
                    break

            if is_close:
                close_bboxes.append(coco_xyxy)

    return close_bboxes


def bbox_overlaps(bbox1: List[float], bbox2: List[float]) -> bool:
    """Check if two bboxes overlap. Both in [x1, y1, x2, y2] format.

    Returns True if rectangles overlap on both x and y axes.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Check if they overlap on both x and y axes
    x_overlap = x1_1 < x2_2 and x2_1 > x1_2
    y_overlap = y1_1 < y2_2 and y2_1 > y1_2

    return x_overlap and y_overlap


def bbox_min_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate minimum distance between two bboxes. Both in [x1, y1, x2, y2] format.

    Returns 0 if bboxes overlap, otherwise returns the minimum distance between
    any point on one bbox to any point on the other bbox (edge-to-edge distance).
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate horizontal distance
    if x2_1 < x1_2:  # bbox1 is to the left of bbox2
        dx = x1_2 - x2_1
    elif x1_1 > x2_2:  # bbox1 is to the right of bbox2
        dx = x1_1 - x2_2
    else:  # bboxes overlap horizontally
        dx = 0

    # Calculate vertical distance
    if y2_1 < y1_2:  # bbox1 is above bbox2
        dy = y1_2 - y2_1
    elif y1_1 > y2_2:  # bbox1 is below bbox2
        dy = y1_1 - y2_2
    else:  # bboxes overlap vertically
        dy = 0

    return (dx**2 + dy**2) ** 0.5


def calculate_encompassing_bbox(bboxes: List[List[float]]) -> List[float]:
    """Calculate bbox that encompasses all given bboxes."""
    if not bboxes:
        return [0, 0, 0, 0]

    # Find min and max coordinates
    min_x1 = min(bbox[0] for bbox in bboxes)
    min_y1 = min(bbox[1] for bbox in bboxes)
    max_x2 = max(bbox[2] for bbox in bboxes)
    max_y2 = max(bbox[3] for bbox in bboxes)

    return [min_x1, min_y1, max_x2, max_y2]


# def get_last_sentence(text: str) -> str:
#     """
#     Extract the last sentence from a text string.
#     Args:
#         text: Input text to parse
#     Returns:
#         The last sentence in the text
#     """
#     if not text.strip():
#         return ""
#     # Split by sentence delimiters (., !, ?)
#     sentences = re.split(r"[.!?]+", text.strip())
#     # Remove empty sentences and get the last non-empty one
#     sentences = [s.strip() for s in sentences if s.strip()]
#     return sentences[-1] if sentences else ""


# def check_last_sentence_for_crop(text: str) -> bool:
#     """
#     Check if the last sentence in the text contains the word "crop".
#     Args:
#         text: Input text to check
#     Returns:
#         True if the last sentence contains "crop" (case-insensitive), False otherwise
#     """
#     last_sentence = get_last_sentence(text)
#     return "crop" in last_sentence.lower()


# def remove_last_sentence_if_has_crop(text: str) -> str:
#     """
#     Remove the last sentence from text if it contains the word "crop".
#     Uses the existing get_last_sentence and check_last_sentence_for_crop functions.
#     Args:
#         text: Input text to process
#     Returns:
#         Text with last sentence removed if it contained "crop", otherwise unchanged
#     """
#     if not text.strip():
#         return text
#     # Check if last sentence has crop using existing function
#     if not check_last_sentence_for_crop(text):
#         return text
#     # Get the last sentence using existing function
#     last_sentence = get_last_sentence(text)
#     if not last_sentence:
#         return text
#     # Find the last occurrence of the sentence and remove it
#     # We'll look for the sentence with possible punctuation
#     # Escape special regex characters in the sentence
#     escaped_sentence = re.escape(last_sentence)
#     # Try to find and remove the sentence (with optional trailing punctuation/whitespace)
#     pattern = escaped_sentence + r"[.!?]*\s*$"
#     # Remove the last sentence
#     result = re.sub(pattern, "", text, count=1)
#     # If that didn't work, try a simpler approach - just remove from the last occurrence
#     if result == text:
#         last_pos = text.rfind(last_sentence)
#         if last_pos != -1:
#             result = text[:last_pos].rstrip()
#     return result.rstrip()


def fix_tool_calling_strings(
    image_id: str, qna_a: str, anno_str: str, coco_annotation_file: Optional[str] = None
) -> str:
    """
    Main function to fix tool-calling strings in qna["A2"].

    Args:
        image_id: The image ID for loading COCO annotations
        qna_a: The A2 string containing tool-calling patterns
        anno_str: The annotation string with correct noun+number patterns
        coco_annotation_file: Optional path to COCO annotation file

    Returns:
        Fixed A2 string
    """
    # print(f"Processing: {image_id}")

    # Load COCO annotations for this image
    coco_ann = load_coco_annotations(image_id, coco_annotation_file)

    # Parse RefCOCO annotations
    annotations = parse_annotations(anno_str)

    # Extract all tool-calling patterns
    tool_calls = extract_tool_calls(qna_a)

    # Process tool calls in reverse order to maintain string positions
    fixed_text = qna_a
    total_fixed = 0
    total_not_found = 0
    total_removed = 0

    for content, start_pos, end_pos in reversed(tool_calls):
        # Check if it needs fixing
        valid_nouns = set(annotations.keys())
        if not is_valid_crop_format(content, valid_nouns):
            print(f"  Needs fixing: {content}")
            fixed_content, not_found, n_fixed, n_not_found = fix_tool_calling_content(
                content, annotations
            )
            bbox = set_bbox(fixed_content, anno_str, annotations, coco_ann)
            if len(bbox) == 0:
                total_removed += 1
                # # Parse last sentence in the text before removal and check for "crop"
                # text_before_removal = fixed_text[:start_pos]
                # last_sentence_has_crop = check_last_sentence_for_crop(text_before_removal)
                # if last_sentence_has_crop:
                #     print(
                #         f"  Last sentence contains 'crop' and is removed: {get_last_sentence(text_before_removal)}"
                #     )
                #     # Remove the crop-related sentence from text before removal
                #     text_before_removal = remove_last_sentence_if_has_crop(text_before_removal)
                # fixed_text = text_before_removal.strip() + fixed_text[end_pos:].strip()
                print(f"  crop removed: {fixed_text[start_pos:end_pos]}")
                fixed_text = fixed_text[:start_pos].strip() + " " + fixed_text[end_pos:].strip()
                continue

            total_fixed += n_fixed
            total_not_found += n_not_found

            if fixed_content != content:
                print(
                    f"  Fixed to: {fixed_content}"
                    + (f", not_found_parts: {not_found}" if not_found else "")
                )
            else:
                print(
                    f"  Could not fix: {content}"
                    + (f", not_found_parts: {not_found}" if not_found else "")
                )

            fixed_content = fixed_content + " " + "[" + ", ".join([f"{x:.2f}" for x in bbox]) + "]"

            # Replace in the text
            fixed_text = fixed_text[: start_pos + 1] + fixed_content + fixed_text[end_pos - 1 :]
        else:
            bbox = set_bbox(content, anno_str, annotations, coco_ann)
            if len(bbox) == 0:
                total_removed += 1
                # Parse last sentence in the text before removal and check for "crop"
                # text_before_removal = fixed_text[:start_pos]
                # last_sentence_has_crop = check_last_sentence_for_crop(text_before_removal)
                # if last_sentence_has_crop:
                #     print(
                #         f"  Last sentence contains 'crop' and is removed: {get_last_sentence(text_before_removal)}"
                #     )
                #     # Remove the crop-related sentence from text before removal
                #     text_before_removal = remove_last_sentence_if_has_crop(text_before_removal)
                # fixed_text = text_before_removal.strip() + fixed_text[end_pos:].strip()
                print(f"  crop removed: {fixed_text[start_pos:end_pos]}")
                fixed_text = fixed_text[:start_pos].strip() + " " + fixed_text[end_pos:].strip()
                continue
            fixed_content = content + " " + "[" + ", ".join([f"{x:.2f}" for x in bbox]) + "]"
            fixed_text = fixed_text[: start_pos + 1] + fixed_content + fixed_text[end_pos - 1 :]

    return fixed_text, total_fixed, total_not_found, total_removed


def fix_tool_calling_strings_verification(
    image_id: str, qna_a: str, anno_str: str, coco_annotation_file: Optional[str] = None
) -> str:
    """
    Main function to fix tool-calling strings in qna["A2"].

    Args:
        image_id: The image ID for loading COCO annotations
        qna_a: The A2 string containing tool-calling patterns
        anno_str: The annotation string with correct noun+number patterns
        coco_annotation_file: Optional path to COCO annotation file

    Returns:
        Fixed A2 string
    """
    # Parse RefCOCO annotations
    annotations = parse_annotations(anno_str)

    # Extract all tool-calling patterns
    tool_calls = extract_tool_calls(qna_a)

    # Process tool calls in reverse order to maintain string positions
    fixed_text = qna_a

    for content, start_pos, end_pos in reversed(tool_calls):
        # Check if it needs fixing
        valid_nouns = set(annotations.keys())
        if not is_valid_crop_format(content, valid_nouns):
            print(f"  Verification found invalid crop: {content}")
            fixed_content, not_found, _, _ = fix_tool_calling_content(content, annotations)

            # if fixed_content != content:
            #     print(f"  Fixed to: {fixed_content}, not_found_parts: {not_found}")
            # else:
            #     print(f"  Could not fix: {content}, not_found_parts: {not_found}")

            # Replace in the text
            fixed_text = fixed_text[: start_pos + 1] + fixed_content + fixed_text[end_pos - 1 :]

    return fixed_text


# Example usage and test
def test_function():
    """Test the function with the provided examples."""

    # Example data
    anno_str_1 = """- person 1 [170.75, 91.57, 425.55, 428.55], who is "a woman standing behind the net", "middle standing up", "standing with racket", "a lady standing up", "person standing u"
- person 2 [114.57, 237.84, 435.4, 655.33], who is "dark - haired woman sitting with leg extended and tennis racquet on lap", "the woman sitting down on the right", "woman sitting with leg extended", "right girl on floor", "ditting darker hair\""""

    qna_a2_1 = """<think>
{Crop woman sitting with tennis racquet on her leg}

The woman sitting with a tennis racket on her lap is smiling.
</think>

<answer>
The woman might be smiling because she's enjoying the moment.
</answer>"""

    print("=== TEST 1: Original Example ===")
    print("Original A2:")
    print(qna_a2_1)
    print("\n" + "-" * 50 + "\n")

    fixed_a2_1, n_fixed, n_not_found = fix_tool_calling_strings("dummy_id", qna_a2_1, anno_str_1)

    print("Fixed A2:")
    print(fixed_a2_1)

    # User's new example - should be more conservative
    anno_str_2 = """- person 1 [103.93, 299.99, 342.08000000000004, 777.4], who is "the back of an older woman with her hair in a barrette with a blue jacket on", "a woman is wearing blue sweater", "the lady with the blue shirt", "woman back in blue", "lady with back to us"
- person 2 [216.58, 261.7, 514.8000000000001, 735.29], who is "the woman in the grey shirt with a watch on her wrist ..", "woman in gray shirt facing camera on right", "woman with gray shirt standing next to man", "a short haired woman in jeans shopping", "gray shirt wearing glasses"
- person 3 [83.6, 270.27, 252.24999999999997, 723.5999999999999], who is "a woman in glasses shops in an open air fruit market", "a woman in a gray coat and scarf"
"""

    qna_a2_2 = """<think>
To determine the type of market the person in the gray coat and scarf is shopping at, I will examine the items visible in the image. The market appears to have various fruits and vegetables, as well as other goods such as baskets and birds in cages. I will zoom in on the fruit and vegetable section to confirm the types of produce available.

{Crop person 3, bananas in front}

Upon closer inspection, there are bananas, oranges, and other fruits prominently displayed, along with a variety of vegetables like cabbages. This suggests a fresh produce market. Additionally, the presence of baskets and birds indicates that this is a general open-air market where various items are sold.

Therefore, the person in the gray coat and scarf is shopping at a fresh produce market.
</think>

<answer>
The person in the gray coat and scarf is shopping at a fresh produce market.
</answer>"""

    print("\n\n=== TEST 2: Conservative Approach (User's Example) ===")
    print("Original A2:")
    print(qna_a2_2)
    print("\n" + "-" * 50 + "\n")

    fixed_a2_2, n_fixed, n_not_found = fix_tool_calling_strings("dummy_id", qna_a2_2, anno_str_2)

    print("Fixed A2:")
    print(fixed_a2_2)

    anno_str_3 = '- person 1 [187.13, 85.22, 530.01, 367.18000000000006], who is "man in front of lamp", "one right of lamp", "far left man", "striped shirt"\n- person 2 [309.9, 143.67, 879.27, 497.29999999999995], who is "person with legs crossed on right", "blue pants man black shirt", "old guy black shirt", "man in black shirt", "laid back man"\n- person 3 [461.44, 178.32, 1085.95, 586.3199999999999], who is "woman in gray with white sleeves", "person on far right", "girl on far right", "child far right", "ponytail girl"\n- person 4 [20.6, 128.36, 167.59, 429.70000000000005], who is "woman siting by herself", "old lady in the chare", "woman in dark shirt", "old lady by herself", "left person"\n- chair 1 [13.17, 139.77, 181.40999999999997, 482.87], which is "a tan sofa chair"'

    qna_a2_3 = "<think>\nFirst, I need to identify the setting where the people in the image are located. I'll start by looking at the background elements and furniture to get a sense of the environment.\n\n{Crop person 1, tan sofa chair}\n\nFrom the cropped image, I can see that person 1 is sitting on a tan sofa chair. The presence of the sofa chair suggests that this is an indoor setting, likely a living room.\n\nNext, I'll examine other details in the room to confirm the setting.\n\n{Crop desk 1, flower arrangement}\n\nThe desk has a flower arrangement, wine glasses, newspapers, and other casual items. This reinforces the idea of a relaxed, domestic environment.\n\nFinally, looking at the overall layout with other furniture and decorations such as the lamp, windows, and other seating arrangements:\n\n{Crop lamp, windows, other seating}\n\nThe combination of the lamp, windows, and additional seating, along with the casual arrangement of people, strongly indicates that this is a living room setting.\n\n</think>\n<answer>\nThe setting where the person is located is a living room.\n</answer>"

    print("\n\n=== TEST 3: ===")
    print("Original A2:")
    print(qna_a2_3)
    print("\n" + "-" * 50 + "\n")

    fixed_a2_3, n_fixed, n_not_found = fix_tool_calling_strings("dummy_id", qna_a2_3, anno_str_3)

    print("Fixed A2:")
    print(fixed_a2_3)

    return fixed_a2_1, fixed_a2_3


# Advanced version with LLM assistance for better matching
async def fix_with_llm_assistance(content: str, annotations: Dict[str, List[str]]) -> str:
    """
    Use a small LLM to help match rephrased descriptions with correct noun+number.

    Example LLM prompt for semantic matching:
    """
    llm_prompt = f"""
    Given these annotations:
    {json.dumps(annotations, indent=2)}
    
    And this description: "{content}"
    
    Which annotation key (noun + number) best matches this description?
    Return only the key (e.g., "person 1") or "none" if no good match.
    """

    # This would make an API call to a small LLM
    # For now, fall back to similarity-based approach
    return fix_tool_calling_content(content, annotations)


# Additional utility functions
def validate_fixed_strings(fixed_text: str, annotations: Dict[str, List[str]]) -> bool:
    """Validate that all tool-calling strings are now in correct format."""
    tool_calls = extract_tool_calls(fixed_text)
    valid_nouns = set(annotations.keys())

    for content, _, _ in tool_calls:
        if not is_valid_crop_format(content, valid_nouns):
            return False
    return True


def get_statistics(original_text: str, fixed_text: str) -> Dict[str, int]:
    """Get statistics about the fixing process."""
    original_calls = extract_tool_calls(original_text)
    fixed_calls = extract_tool_calls(fixed_text)

    changes_made = 0
    for i, (orig, _, _) in enumerate(original_calls):
        if i < len(fixed_calls) and orig != fixed_calls[i][0]:
            changes_made += 1

    return {
        "total_tool_calls": len(original_calls),
        "changes_made": changes_made,
        "success_rate": (len(original_calls) - changes_made) / len(original_calls)
        if original_calls
        else 1.0,
    }


def test_set_bbox():
    # Test edge cases
    # Test set_bbox function specifically
    print("\n" + "=" * 60)
    print("TESTING set_bbox FUNCTION")
    print("=" * 60)

    # Test case for set_bbox
    test_image_id = "000000575594"
    test_fixed_content = "Crop teddy bear in package with beach ball"
    test_anno_str = '- teddy bear 1 [384.78, 96.91, 605.48, 341.60], which is "teddy bear in package with beach ball", "bear with blue and yellow ball", "teady bear on the right", "blueyellow beach ball", "bear with the ball"\n- teddy bear 2 [79.64, 88.28, 307.06, 334.88], which is "a small teddy bear in a pink tutu", "the stuffed bear wearing a tutu", "left bear dressed in pink", "bear on left"'
    test_annotations = parse_annotations(test_anno_str)
    test_coco_ann = load_coco_annotations(test_image_id)

    print(f"Testing set_bbox with:")
    print(f"  fixed_content: {test_fixed_content}")
    print(f"  Known objects in annotations: {list(test_annotations.keys())}")
    print(f"  COCO annotations available: {len(test_coco_ann.get('annotations', []))}")

    result_bbox = set_bbox(test_fixed_content, test_anno_str, test_annotations, test_coco_ann)
    print(f"  Result bbox: {result_bbox}")
    print("\n" + "=" * 60)
    print("TESTING set_bbox FUNCTION")
    print("=" * 60)

    # Test case for set_bbox
    test_image_id = "000000575055"
    test_fixed_content = "Crop striped chair with pink cushion, coffee table"
    test_anno_str = '- chair 1 [509.00, 215.86, 640.00, 372.94], which is "right most chair kinda dark", "right chair facing us", "striped chair by the piano", "chair closest to bench", "sofa dark striped"\n- couch 1 [476.83, 402.93, 638.39, 495.00], which is "couch arm that is barely on the screen corner", "least visible couch arm with pink on it", "couch arm bottom right", "sofa on right cut off", "partial arm at 4 o clock"\n- couch 2 [509.97, 215.46, 640.00, 367.34], which is "a striped couch in the shade with a pink pillow on it", "the sriped seat in the shadows facing us", "striped chair furthest right", "chair on other side of table", "chair on right side"\n- couch 3 [167.13, 265.06, 449.23, 495.00], which is "a striped sofa chair with pink cushion which is a part of full sofa set and a table in a sitting room", "striped chair with pink pillow on it", "striped chair middle", "couch on the left", "upholstered chair"\n- chair 2 [169.26, 267.19, 449.23, 489.68], which is "striped seat with back showing", "hideously ugly white chair", "chair front center", "striped couch", "near chair"'
    test_annotations = parse_annotations(test_anno_str)
    test_coco_ann = load_coco_annotations(test_image_id)

    print(f"Testing set_bbox with:")
    print(f"  fixed_content: {test_fixed_content}")
    print(f"  Known objects in annotations: {list(test_annotations.keys())}")
    print(f"  COCO annotations available: {len(test_coco_ann.get('annotations', []))}")

    result_bbox = set_bbox(test_fixed_content, test_anno_str, test_annotations, test_coco_ann)
    print(f"  Result bbox: {result_bbox}")
    print("\n" + "=" * 60)
    print("TESTING set_bbox FUNCTION")
    print("=" * 60)

    # Test case for set_bbox
    test_image_id = "000000575055"
    test_fixed_content = "Crop person 3, cake"
    test_anno_str = '- person 1 [161.79, 0.74, 555.38, 194.65], who is "a person in green top taking a photo", "a woman holding a camera", "person holding camera", "green shirt person", "green shirt"\n- person 2 [497.45, 3.74, 1137.45, 179.53000000000003], who is "person in striped shirt in upper right corner", "person in the tan and white striped shirt", "stripe shirt in upper right hand corner", "person right corner", "striped shirt no head"\n- person 3 [0.0, 14.22, 463.41, 494.22], who is "a young blonde woman wearing a pink bachelorette sash , cuts a cake", "a women in pink cutting cake", "bachelorette party girl", "girl in pink", "pink shirt"'
    test_annotations = parse_annotations(test_anno_str)
    test_coco_ann = load_coco_annotations(test_image_id)

    print(f"Testing set_bbox with:")
    print(f"  fixed_content: {test_fixed_content}")
    print(f"  Known objects in annotations: {list(test_annotations.keys())}")
    print(f"  COCO annotations available: {len(test_coco_ann.get('annotations', []))}")

    result_bbox = set_bbox(test_fixed_content, test_anno_str, test_annotations, test_coco_ann)
    print(f"  Result bbox: {result_bbox}")
    print("\n" + "=" * 60)
    print("TESTING set_bbox FUNCTION")
    print("=" * 60)

    # Test case for set_bbox
    test_image_id = "000000575049"
    test_fixed_content = "Crop bookshelf, plant"
    test_anno_str = '- person 1 [548.88, 204.48, 1188.88, 675.87], who is "a man with a black shirt on wearing glasses", "man on right half in frame", "th guy oni the far right", "only half of him is visable", "far right guy cut off"\n- person 2 [194.8, 54.46, 782.4200000000001, 529.72], who is "fully visible man in a black t - shirt and holding a wii controller", "a boy with black t - shirt holding a remote and smilling", "guy in black holding controller", "full man in black", "man with remote"\n- person 3 [0.0, 19.63, 374.13, 494.59], who is "a woman wearing glasses and a blue sweater", "a smiling young woman with long hair and glasses", "love c lloyd lol lady", "blue shirt", "woman"'
    test_annotations = parse_annotations(test_anno_str)
    test_coco_ann = load_coco_annotations(test_image_id)

    print(f"Testing set_bbox with:")
    print(f"  fixed_content: {test_fixed_content}")
    print(f"  Known objects in annotations: {list(test_annotations.keys())}")
    print(f"  COCO annotations available: {len(test_coco_ann.get('annotations', []))}")

    result_bbox = set_bbox(test_fixed_content, test_anno_str, test_annotations, test_coco_ann)
    print(f"  Result bbox: {result_bbox}")
    print("\n" + "=" * 60)
    print("TESTING set_bbox FUNCTION")
    print("=" * 60)

    # Test case for set_bbox
    test_image_id = "000000577405"
    test_fixed_content = "Crop orange 2, apple"
    test_anno_str = '- orange 1 [81.04, 199.01, 302.16, 561.21], which is "the leftmost orange"\n- orange 2 [166.02, 215.72, 530.82, 613.27], which is "orange closest to apple"'
    test_annotations = parse_annotations(test_anno_str)
    test_coco_ann = load_coco_annotations(test_image_id)

    print(f"Testing set_bbox with:")
    print(f"  fixed_content: {test_fixed_content}")
    print(f"  Known objects in annotations: {list(test_annotations.keys())}")
    print(f"  COCO annotations available: {len(test_coco_ann.get('annotations', []))}")

    result_bbox = set_bbox(test_fixed_content, test_anno_str, test_annotations, test_coco_ann)
    print(f"  Result bbox: {result_bbox}")
    print("\n" + "=" * 60)
    print("TESTING set_bbox FUNCTION")
    print("=" * 60)

    # Test case for set_bbox
    test_image_id = "000000575421"
    test_fixed_content = "Crop person 1, pizzas"
    test_anno_str = '- pizza 1 [1.08, 87.37, 641.08, 398.02], which is "trays of pizza , with pepperoni , black olive , and cheddar cheese topping", "smaller pizza with piece missing", "cut pizza on top left", "top left pizza", "the pepperoni pizza"\n- pizza 2 [64.72, 291.24, 544.72, 763.69], which is "pizza in the front with slice missing", "a pizza with one slice taken from it", "pizza with one small piece missing", "pizza sausage olives close to us", "bottom left pizza"\n- person 1 [461.05, 0.0, 1101.05, 183.9], who is "black pants of person standing to side of pizzas", "the person standing next to the pizzas"'
    test_annotations = parse_annotations(test_anno_str)
    test_coco_ann = load_coco_annotations(test_image_id)

    print(f"Testing set_bbox with:")
    print(f"  fixed_content: {test_fixed_content}")
    print(f"  Known objects in annotations: {list(test_annotations.keys())}")
    print(f"  COCO annotations available: {len(test_coco_ann.get('annotations', []))}")

    result_bbox = set_bbox(test_fixed_content, test_anno_str, test_annotations, test_coco_ann)
    print(f"  Result bbox: {result_bbox}")


if __name__ == "__main__":
    # Run comprehensive tests
    # result = test_function()
    # test_set_bbox()

    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    example_image_id = "000000581227"

    anno_str = '- person 1 [42.46, 120.08, 125.90, 333.00], who is \"the woman at the rear of the court\", \"person in background\", \"far left player\", \"person with glasses\", \"player not in focus\"\n- person 2 [263.09, 13.34, 426.96, 329.05], who is \"a woman with black hair holding the tennis racket in front\", \"the woman on the right\", \"girl in front cam\", \"girl closest to us\", \"right girl\"\n- tennis racket 1 [373.12, 103.18, 500.00, 246.76], which is \"a tennis racket being held by a girl who is not wearing sunglasses\", \"a black tennis racket with white markings\"'

    test_a3 = "<think>\nFirst, I need to identify the person holding the tennis racket in the foreground. Looking at the image, I notice two women on the tennis court. The woman in the foreground is holding a tennis racket. \n\nTo confirm, I'll use the \"Crop\" tool to zoom in on the area where the tennis racket is being held. This will help me clearly see who is holding it.\n\n{Crop person 2, tennis racket 1 [263.09, 13.34, 500.00, 329.05]}\n\nUpon closer inspection, I can see that the woman in the foreground, who has black hair, is indeed holding the tennis racket. She is the one in focus and is the closest to the camera.\n\nTherefore, the answer should be the woman with black hair holding the tennis racket in front.\n</think>\n\n<answer>The woman with black hair holding the tennis racket in front is holding the tennis racket in the foreground.</answer>"

    print("Multi-entity test:")
    print("Original:", test_a3.strip())
    fixed_multi, n_fixed, n_not_found, total_removed = fix_tool_calling_strings_verification(
        example_image_id, test_a3, anno_str
    )
    print("Fixed:", fixed_multi.strip())
    print(f"Items fixed: {n_fixed}, Items not found: {n_not_found}")

    example_image_id = "000000477580"

    anno_str = '- bottle 1 [305.68, 205.70, 545.81, 287.13], which is "a bottle of wine laying down amongst the veggies", "a bottle of wine between the vegetables", "wine bottle laying down", "made of glass laying", "sideways wine bottle"\n- bottle 2 [234.57, 0.77, 340.81, 261.31], which is "a bottle of red wine , labeled chocovine , which has a picture of a windmill on it", "a bottle of wine which is kept in the middle of vegetables", "the bottle of choco vine", "wine bottle stand up", "standing bottle"\n- carrot 1 [199.00, 358.00, 460.00, 419.00], which is "the carrot on the very bottom"\n- broccoli 1 [0.00, 215.08, 158.26, 351.44], which is "a group of broccoli", "a head of broccoli sits on the table"\n- carrot 2 [18.00, 277.00, 236.00, 419.00], which is "the carrot to the left of the broccol", "a carrot going across the others"\n- carrot 3 [111.93, 348.74, 409.29, 395.92], which is "a carrot stacked in the middle of two other carrots", "the middle carrot that \' s laying down with two other carrots , on either side"'

    test_a2 = "<think>\nTo determine the wine bottle's position in relation to the vegetables, I need to closely examine their arrangement in the image. I will start by using the crop tool to zoom in on the wine bottles and the surrounding vegetables.\n\n{Crop bottle 1, bottle 2, carrot 1, carrot 2, carrot 3, broccoli 1}\n\nUpon inspecting the cropped region, I can see that bottle 1 is laying down among the vegetables, particularly close to the carrots and broccoli. Bottle 2 is standing upright and is positioned between the vegetables, with a more central placement compared to bottle 1.\n\nThe positioning suggests that the wine bottles are intentionally placed among the vegetables, possibly as part of a display or to create an aesthetically pleasing arrangement. Bottle 1's horizontal position indicates it might be part of a laid-out setting, while bottle 2's vertical position makes it a focal point within the vegetable arrangement.\n\nOverall, the wine bottles are integrated with the vegetables, enhancing the visual appeal of the setup.\n</think>\n\n<answer>\nThe wine bottles are intentionally placed among the vegetables to create an aesthetically pleasing arrangement. Bottle 1 is laying down among the carrots and broccoli, while bottle 2 stands upright and serves as a focal point within the vegetable display.\n</answer>"

    print("Multi-entity test:")
    print("Original:", test_a2.strip())
    fixed_multi, n_fixed, n_not_found, total_removed = fix_tool_calling_strings(
        example_image_id, test_a2, anno_str
    )
    print("Fixed:", fixed_multi.strip())
    print(f"Items fixed: {n_fixed}, Items not found: {n_not_found}")

    example_image_id = "000000577405"

    anno_str = '- orange 1 [81.04, 199.01, 302.16, 561.21], which is "the leftmost orange in the middle of a white plate with 3 other oranges , an apple , and a muffin setting on top of the fruit", "partial orange second from bottom on left", "orange with a light shining in half white line", "orange in middle on the left", "orange with horizontal light stripe"\n- orange 2 [166.02, 215.72, 530.82, 613.27], which is "an orange that is next to an apple and another orange and it is in front", "orange closest to you next to apple", "orange bottom front", "orange in front of bowl", "orange nearest you"'

    test_a2 = "<think>\nTo determine the position of the orange closest to the apple, I first need to identify the apple on the plate. The apple is located towards the top right corner of the plate. There are several oranges on the plate, but I need to find the one nearest to the apple.\n\nBy visually inspecting the arrangement of the fruits, I notice that there is an orange directly below the apple. This orange appears to be the closest one to the apple. To confirm this observation, I will use the crop tool to zoom in on this specific area and inspect the spatial relationship more closely.\n\n{Crop orange closest to apple, apple}\n\nAfter zooming in, it's clear that the orange directly below the apple is indeed the closest one. This orange is positioned to the bottom left of the apple on the plate.\n\n</think>\n<answer>The orange closest to the apple on the plate is positioned directly below the apple.</answer>"

    print("Multi-entity test:")
    print("Original:", test_a2.strip())
    fixed_multi, n_fixed, n_not_found, total_removed = fix_tool_calling_strings(
        example_image_id, test_a2, anno_str
    )
    print("Fixed:", fixed_multi.strip())
    print(f"Items fixed: {n_fixed}, Items not found: {n_not_found}")

    anno_str = """- person 1 [170.75, 91.57, 425.55, 428.55], who is "a woman standing behind the net", "middle standing up", "standing with racket", "a lady standing up", "person standing u"
- person 2 [114.57, 237.84, 435.4, 655.33], who is "dark - haired woman sitting with leg extended and tennis racquet on lap", "the woman sitting down on the right", "woman sitting with leg extended", "right girl on floor", "ditting darker hair\""""

    # Test multiple entities
    test_multi = """<think>
{Crop person 1, dark - haired woman sitting with leg extended and tennis racquet on lap}
</think>"""

    test_multi = """<think>\nTo understand the relationship between the woman standing and the two seated women, let's analyze the image. \n\nFirst, the woman standing is holding a tennis racket and is positioned behind the net, which might indicate she is ready to play or has just finished playing. The two seated women are also holding tennis rackets and are dressed in tennis attire. Their relaxed posture suggests they might be taking a break.\n\nGiven their similar attire and equipment, it's possible they are part of the same group, perhaps friends or teammates. The setting appears to be a casual tennis match or practice session.\n\nTo get a closer look at their expressions and possible interactions, let's crop the image to focus on their faces and upper bodies.\n\n{Crop person 1, right woman}\n</think>\n\n<think>\nLooking at the cropped image, the expressions of the women suggest familiarity and comfort with each other. The woman standing appears to be in a relaxed stance, and the seated women are smiling, which could indicate a friendly relationship.\n\nConsidering the context of a tennis setting, it's plausible that these women are friends or teammates enjoying a game together. Their similar attire and the shared activity support this idea.\n\nTherefore, the likely relationship between the woman standing and the two seated women is that they are friends or teammates engaged in a tennis activity.\n</think>\n\n<answer>\nThe likely relationship between the woman standing and the two seated women is that they are friends or teammates.\n</answer>"""

    print("Multi-entity test:")
    print("Original:", test_multi.strip())
    fixed_multi, n_fixed, n_not_found, total_removed = fix_tool_calling_strings(
        "dummy_id", test_multi, anno_str
    )
    print("Fixed:", fixed_multi.strip())

    # Test non-existent entity
    test_nonexistent = """<think>
{Crop person 1, banana}
</think>"""

    print("\nNon-existent entity test:")
    print("Original:", test_nonexistent.strip())
    fixed_nonexistent, n_fixed, n_not_found, total_removed = fix_tool_calling_strings(
        "dummy_id", test_nonexistent, anno_str
    )
    print("Fixed:", fixed_nonexistent.strip())

    # Get statistics
    annotations = parse_annotations(anno_str)
    stats = get_statistics(test_multi, fixed_multi)
    print(f"\nStatistics: {stats}")
    print(f"Validation passed: {validate_fixed_strings(fixed_multi, annotations)}")
