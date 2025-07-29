"""
RefCOCO Dataset Processing with Intelligent Referring Expression Merging

Key Insight: RefCOCO, RefCOCOplus, and RefCOCOg all use the same underlying COCO annotations,
so bounding boxes are identical. The real value is in merging diverse referring expressions
from different datasets to create richer, more comprehensive descriptions.

This approach:
1. Uses proper COCO category mapping via category_id
2. Merges referring expressions from overlapping annotations (same bbox + category)
3. Applies semantic deduplication and quality scoring to descriptions
4. Generates enhanced questions and multimodal reasoning responses
"""
import os
import json
import torch
from datasets import load_dataset
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Your InternVL3 imports
from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor import load_models, split_model

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
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
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
            spatial_words = ['left', 'right', 'front', 'back', 'center', 'middle', 'top', 'bottom']
            score += sum(5 for word in spatial_words if word in desc.lower())
            
            # Bonus for color information
            color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'pink']
            score += sum(3 for word in color_words if word in desc.lower())
            
            # Bonus for action/state information
            action_words = ['sitting', 'standing', 'walking', 'holding', 'wearing', 'looking']
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

class RefCOCOProcessor:
    def __init__(self, model_path="OpenGVLab/InternVL3-78B"):
        self.dataset_p_root = "/mnt/data/"
        print("Initializing model...")
        device_map = split_model(model_path)
        self.model, self.tokenizer = load_models(model_path, device_map)
        self.description_merger = DescriptionMerger()
        
        self.question_2_comm = """<image>
##Object names with descriptions
{anno}
##
You are performing "Multimodal Interleaved Reasoning". During the thinking process, you need to keep an eye on the visual cues in the original image, find regions of the image that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{{Crop (object name)}}
Ensure that you "Crop" at least once. If you crop the region including multiple objects, list them with commas. For example, {{Crop person 1, desk 3}}, {{Crop person 2}}.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags. Do not use numbered object name outside the crop tool, but use noun phrase with description instead.
Question: {question}"""
        
        self.question_2_followup = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer"
    
    def get_image_id_from_path(self, image_path):
        """Extract image ID from path for deduplication"""
        # Example: coco/train2017/000000549347.jpg -> 549347
        # import re
        # match = re.search(r'COCO_\w+_(\d+)', image_path)
        match = re.search(r'(\d+)\.jpg', image_path)
        return match.group(1) if match else image_path
    
    def load_datasets(self):
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
                    image_path = image_path.replace('train2014/COCO_train2014_', 'train2017/') # Temporary code because of the local foldername issue.
                    image_id = self.get_image_id_from_path(image_path)
                    
                    # Get category from category_id using COCO mapping
                    category_id = sample.get('category_id')
                    category = COCO_CATEGORIES.get(category_id, 'object')
                    
                    bbox = xywh_to_xyxy(sample['bbox'])
                    descriptions = sample['captions'] if isinstance(sample['captions'], list) else [sample['captions']]
                    
                    annotation = {
                        'category': category,
                        'category_id': category_id,
                        'bbox': bbox,
                        'descriptions': descriptions,
                        'dataset_source': dataset_name,
                        'image_path': image_path
                    }
                    
                    image_data[image_id]['annotations'].append(annotation)
                    # Keep track of the image path (use the first one encountered)
                    if 'image_path' not in image_data[image_id]:
                        image_data[image_id]['image_path'] = image_path
        
        print(f"Found {len(image_data)} unique images across all datasets")
        
        # Second pass: merge identical annotations and create final structure
        final_data = []
        
        for image_id, data in tqdm(image_data.items(), desc="Merging identical annotations"):
            merged_annotations = self.merge_identical_annotations(data['annotations'])
            
            if merged_annotations:  # Only include images with valid annotations
                anno_string = self.create_anno_string(merged_annotations)
                
                data_entry = {
                    "image_path": data['image_path'],
                    "image_id": image_id,
                    "anno": anno_string,
                    "annotations": merged_annotations,
                    "questions": [],
                    "responses_1": [],
                    "responses_2": []
                }
                final_data.append(data_entry)
        
        print(f"Final dataset contains {len(final_data)} unique images with merged referring expressions")
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
            
            for j, ann2 in enumerate(annotations[i+1:], i+1):
                if used[j] or ann1['category'] != ann2['category']:
                    continue
                
                # Check if bboxes are nearly identical (allowing small tolerance for floating point differences)
                bbox_diff = [abs(ann1['bbox'][k] - ann2['bbox'][k]) for k in range(4)]
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
        all_description_lists = [ann['descriptions'] for ann in annotations]
        
        # Merge descriptions intelligently
        merged_descriptions = self.description_merger.merge_descriptions(all_description_lists)
        
        # Use the first bbox (they should be nearly identical across datasets)
        bbox = base_ann['bbox']
        
        # Collect source datasets
        sources = list(set(ann['dataset_source'] for ann in annotations))
        
        return {
            'category': base_ann['category'],
            'category_id': base_ann['category_id'],
            'bbox': bbox,  # Keep original bbox since they should be identical
            'descriptions': merged_descriptions,
            'dataset_sources': sources,
            'merged_from': len(annotations)
        }
    
    def create_anno_string(self, annotations):
        """Create annotation string with improved descriptions"""
        anno_parts = []
        object_counter = defaultdict(int)
        
        for ann in annotations:
            category = ann['category']
            object_counter[category] += 1
            obj_num = object_counter[category]
            
            bbox_str = f"[{ann['bbox'][0]}, {ann['bbox'][1]}, {ann['bbox'][2]}, {ann['bbox'][3]}]"
            
            # Use merged, high-quality descriptions
            descriptions = ', '.join([f'"{desc}"' for desc in ann['descriptions']])
            connector = "who is" if category == "person" else "which is"
            
            # Add source info for quality tracking (optional)
            source_info = f" (merged from {ann.get('merged_from', 1)} sources)" if ann.get('merged_from', 1) > 1 else ""
            
            anno_parts.append(f"- {category} {obj_num} {bbox_str}, {connector} {descriptions}")
        
        return "\n".join(anno_parts)
    
    def generate_initial_questions(self, data_list):
        """Generate initial questions for each object"""
        print("Generating initial questions...")
        
        for data_entry in tqdm(data_list, desc="Processing images"):
            image_path = data_entry["image_path"]
            anno = data_entry["anno"]
            
            try:
                pixel_values = load_image(os.path.join(self.dataset_p_root, image_path), max_num=12).to(torch.bfloat16).cuda()
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
            
            # Generate questions for each object
            object_counter = defaultdict(int)
            all_questions = []
            all_answers = []
            
            for ann in data_entry["annotations"]:
                category = ann['category']
                object_counter[category] += 1
                obj_num = object_counter[category]
                
                bbox_str = f"[{ann['bbox'][0]}, {ann['bbox'][1]}, {ann['bbox'][2]}, {ann['bbox'][3]}]"
                
                # Enhanced question prompt that mentions the quality of merged referring expressions
                question_prompt = f"""<image>
Image has these objects with bboxes and descriptions:
{anno}
Create two or three questions from the visuals of the {category} {obj_num} {bbox_str}. Generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly mention numberred object name or the presence of the descriptions in your response.
Output format: {{Question: ...\nReasoning: ...\nAnswer: ...}}"""
                
                try:
                    response = self.model.chat(self.tokenizer, pixel_values, question_prompt, generation_config)
                    questions = self.extract_questions_n_answers_from_response(response, lookfor='Question')
                    answers = self.extract_questions_n_answers_from_response(response, lookfor='Answer')
                    all_questions.extend(questions)
                    all_answers.extend(answers)
                except Exception as e:
                    print(f"Error generating questions: {e}")
                    continue
            
            data_entry["questions"] = all_questions
            data_entry["responses_1"] = all_answers
        
        return data_list
    
    def extract_questions_n_answers_from_response(self, response, lookfor="Question"):
        """Extract individual questions from model response"""
        # Primary pattern: Looks for "Question:" on one line and the question on the next.
        pattern = rf"^(?:#+\s*)?{re.escape(lookfor)}.*?\n(.*?)$"
        matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
        questions = [match.strip() for match in matches if match.strip()]

        # If the primary pattern doesn't find anything, use fallbacks.
        if not questions:
            # Fallback 1: Look for lines containing "question" and a colon.
            fb1_questions = [line.strip()[line.find(':') + 1:].strip()
                             for line in response.split('\n')
                             if lookfor.lower() in line.lower()]
            questions = [q for q in fb1_questions if q]
        
        if not questions:
            # Fallback 2: Look for numbered list questions (e.g., "1. What is...")
            pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                questions = [match.strip() for match in matches]

        if not questions:
            # Fallback 3: Look for any line with a question mark.
            lines = [line.strip() for line in response.split('\n')
                     if line.strip() and not line.strip().startswith('#') and '?' in line]
            questions = lines[:3]  # Limit to 3 questions

        return questions
    
    def generate_detailed_responses(self, data_list):
        """Generate detailed responses with multimodal reasoning"""
        print("Generating detailed responses...")
        
        for data_entry in tqdm(data_list, desc="Detailed responses"):
            image_path = data_entry["image_path"]
            anno = data_entry["anno"]
            
            try:
                pixel_values = load_image(os.path.join(self.dataset_p_root, image_path), max_num=12).to(torch.bfloat16).cuda()
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
            
            detailed_responses = []
            
            for question in data_entry["questions"]:
                if not question.strip():
                    continue
                
                try:
                    # Remove bbox from annotation string for this prompt
                    anno_for_prompt = re.sub(r' \[[^\]]*\]', '', anno)
                    # Initial detailed response
                    prompt_1 = self.question_2_comm.format(anno=anno_for_prompt, question=question)
                    response_1, history = self.model.chat(
                        self.tokenizer, pixel_values, prompt_1, generation_config, return_history=True
                    )
                    
                    # Follow-up refinement
                    response_2 = self.model.chat(self.tokenizer, pixel_values, self.question_2_followup, generation_config, history=history)
                    
                    detailed_responses.append({
                        "question": question,
                        "initial_response": response_1,
                        "final_response": response_2
                    })
                    
                except Exception as e:
                    print(f"Error in detailed response for '{question}': {e}")
                    continue
            
            data_entry["responses_2"] = detailed_responses
        
        return data_list
    
    def save_results(self, data_list, output_prefix="refcoco_merged_results"):
        """Save results with merge statistics"""
        print("Saving results...")
        
        # Calculate merge statistics
        total_merged = sum(1 for entry in data_list for ann in entry["annotations"] if ann.get('merged_from', 1) > 1)
        total_annotations = sum(len(entry["annotations"]) for entry in data_list)
        
        print(f"Merge Statistics:")
        print(f"Total objects: {total_annotations}")
        print(f"Objects with merged referring expressions: {total_merged}")
        print(f"Merge rate: {total_merged/total_annotations*100:.1f}%")
        
        # JSON format with merge info
        json_output = []
        for entry in data_list:
            json_entry = {
                "image_path": entry["image_path"],
                "image_id": entry["image_id"],
                "anno": entry["anno"],
                "merge_info": {
                    "total_objects": len(entry["annotations"]),
                    "objects_with_merged_expressions": sum(1 for ann in entry["annotations"] if ann.get('merged_from', 1) > 1),
                    "dataset_sources": list(set(source for ann in entry["annotations"] 
                                               for source in ann.get('dataset_sources', [])))
                },
                "questions_and_responses": entry["responses_2"]
            }
            json_output.append(json_entry)
        
        with open(f"{output_prefix}.json", "w", encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Python format
        with open(f"{output_prefix}.py", "w", encoding='utf-8') as f:
            f.write("# RefCOCO dataset with intelligent merging of referring expressions\n")
            f.write(f"# Merge statistics: {total_merged}/{total_annotations} objects have merged referring expressions\n\n")
            f.write("data_refcoco_merged = [\n")
            
            for i, entry in enumerate(json_output):
                f.write("    {\n")
                f.write(f"        \"image_path\": \"{entry['image_path']}\",\n")
                f.write(f"        \"image_id\": \"{entry['image_id']}\",\n")
                f.write(f"        \"anno\": \"\"\"\n{entry['anno']}\n\"\"\",\n")
                f.write(f"        \"merge_info\": {entry['merge_info']},\n")
                f.write("        \"questions\": [")
                
                questions = [qr["question"] for qr in entry["questions_and_responses"]]
                f.write(", ".join([f'"{q}"' for q in questions]))
                f.write("],\n")
                
                f.write("        \"responses\": [\n")
                for qr in entry["questions_and_responses"]:
                    f.write(f"            \"\"\"{qr['final_response']}\"\"\",\n")
                f.write("        ]\n")
                
                f.write("    },\n" if i < len(json_output) - 1 else "    }\n")
            
            f.write("]\n")
        
        print(f"Results saved to {output_prefix}.json and {output_prefix}.py")
        return json_output

def main():
    """
    Main execution function
    
    Key insight: RefCOCO datasets share the same COCO annotations (identical bboxes),
    but have different referring expressions. We merge these expressions rather than bboxes.
    """
    processor = RefCOCOProcessor()
    
    # Load datasets with intelligent merging of referring expressions
    data_list = processor.load_datasets()
    print(f"Loaded {len(data_list)} unique images with merged referring expressions")
    
    # Generate initial questions
    data_list = processor.generate_initial_questions(data_list)
    
    # Generate detailed responses
    data_list = processor.generate_detailed_responses(data_list)

    # Generate object_localization questions and responses
    
    # Save results
    results = processor.save_results(data_list)
    
    # Print final statistics
    total_questions = sum(len(entry["questions_and_responses"]) for entry in results)
    print(f"\nFinal Statistics:")
    print(f"Unique images: {len(results)}")
    print(f"Total questions: {total_questions}")
    print(f"Average questions per image: {total_questions/len(results):.2f}")

if __name__ == "__main__":
    main()