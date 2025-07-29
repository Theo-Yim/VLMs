import json
import torch
from datasets import load_dataset
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Set

# Your InternVL3 imports
from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor import load_models, split_model

# COCO Category Mapping (80 classes + background)
COCO_CATEGORIES = {
    0: '__background__',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
    25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
    30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
    35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard',
    39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork',
    44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple',
    49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog',
    54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch',
    59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv',
    64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone',
    69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator',
    74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear',
    79: 'hair drier', 80: 'toothbrush'
}

def xywh_to_xyxy(box):
    """Convert xywh format to xyxy format"""
    bbox = deepcopy(box)
    bbox[2] = box[0] + box[2]
    bbox[3] = box[1] + box[3]
    return bbox

def normalize_bbox(bbox: List[float], tolerance: float = 1.0) -> Tuple[int, int, int, int]:
    """Normalize bbox coordinates for duplicate detection with tolerance"""
    return (
        round(bbox[0] / tolerance) * tolerance,
        round(bbox[1] / tolerance) * tolerance, 
        round(bbox[2] / tolerance) * tolerance,
        round(bbox[3] / tolerance) * tolerance
    )

def create_duplicate_key(image_path: str, bbox: List[float]) -> Tuple[str, Tuple[float, float, float, float]]:
    """Create a key for duplicate detection"""
    normalized_bbox = normalize_bbox(bbox)
    return (image_path, tuple(normalized_bbox))

class ReferringExpressionMerger:
    """Class to handle merging and quality enhancement of referring expressions"""
    
    def __init__(self):
        self.stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    
    def clean_expression(self, expr: str) -> str:
        """Clean and normalize referring expression"""
        # Remove extra whitespace and convert to lowercase for comparison
        cleaned = ' '.join(expr.strip().split())
        return cleaned
    
    def calculate_specificity_score(self, expr: str) -> float:
        """Calculate specificity score for ranking expressions"""
        words = expr.lower().split()
        # Longer expressions with more descriptive words get higher scores
        word_count = len(words)
        descriptive_words = len([w for w in words if w not in self.stop_words and len(w) > 2])
        
        # Bonus for spatial relationships
        spatial_bonus = 1.2 if any(spatial in expr.lower() for spatial in 
                                 ['left', 'right', 'front', 'back', 'top', 'bottom', 'center', 'middle']) else 1.0
        
        # Bonus for specific descriptors
        descriptor_bonus = 1.1 if any(desc in expr.lower() for desc in 
                                    ['wearing', 'holding', 'sitting', 'standing', 'looking', 'facing']) else 1.0
        
        return (word_count + descriptive_words * 2) * spatial_bonus * descriptor_bonus
    
    def merge_expressions(self, expressions: List[str]) -> List[str]:
        """Merge and rank referring expressions"""
        # Clean expressions
        cleaned_expressions = [self.clean_expression(expr) for expr in expressions]
        
        # Remove exact duplicates while preserving order
        seen = set()
        unique_expressions = []
        for expr in cleaned_expressions:
            expr_lower = expr.lower()
            if expr_lower not in seen:
                seen.add(expr_lower)
                unique_expressions.append(expr)
        
        # Remove very similar expressions (fuzzy matching)
        filtered_expressions = self.remove_similar_expressions(unique_expressions)
        
        # Score and rank expressions
        scored_expressions = [(expr, self.calculate_specificity_score(expr)) 
                            for expr in filtered_expressions]
        scored_expressions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top expressions (limit to avoid overly long annotations)
        return [expr for expr, _ in scored_expressions[:6]]  # Keep top 6 expressions
    
    def remove_similar_expressions(self, expressions: List[str]) -> List[str]:
        """Remove very similar expressions using simple similarity check"""
        filtered = []
        
        for expr in expressions:
            is_similar = False
            expr_words = set(expr.lower().split())
            
            for existing in filtered:
                existing_words = set(existing.lower().split())
                # If expressions share >80% of words, consider them similar
                overlap = len(expr_words & existing_words)
                union = len(expr_words | existing_words)
                if union > 0 and overlap / union > 0.8:
                    is_similar = True
                    break
            
            if not is_similar:
                filtered.append(expr)
        
        return filtered

class RefCOCOProcessor:
    def __init__(self, model_path="OpenGVLab/InternVL3-78B"):
        print("Initializing model...")
        device_map = split_model(model_path)
        self.model, self.tokenizer = load_models(model_path, device_map)
        self.expression_merger = ReferringExpressionMerger()
        
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
    
    def load_and_merge_datasets(self):
        """Load RefCOCO datasets and merge duplicates"""
        data_path_lst = [
            "jxu124/RefCOCO",
            "jxu124/RefCOCOplus",
            "jxu124/RefCOCOg",
        ]
        
        # Dictionary to collect all annotations by (image_path, bbox) key
        merged_annotations = defaultdict(lambda: {
            'image_path': '',
            'category_id': None,
            'bbox': None,
            'expressions': [],
            'dataset_sources': []
        })
        
        print("Loading and merging datasets...")
        for data_path in data_path_lst:
            print(f"Processing dataset: {data_path}")
            dataset = load_dataset(data_path)
            
            dataset_name = data_path.split('/')[-1]  # RefCOCO, RefCOCOplus, RefCOCOg
            
            for split in dataset.keys():
                for sample in tqdm(dataset[split], desc=f"Processing {data_path}/{split}"):
                    image_path = sample["image_path"]
                    bbox = sample['bbox']
                    category_id = sample.get('category_id', 1)  # Default to 'person' if missing
                    
                    # Create duplicate detection key
                    dup_key = create_duplicate_key(image_path, xywh_to_xyxy(bbox))
                    
                    # Get expressions (handle both list and single string)
                    expressions = sample['answer'] if isinstance(sample['answer'], list) else [sample['answer']]
                    
                    # Merge with existing entry or create new one
                    entry = merged_annotations[dup_key]
                    if not entry['image_path']:  # New entry
                        entry['image_path'] = image_path
                        entry['category_id'] = category_id
                        entry['bbox'] = bbox
                    
                    # Add expressions and track source datasets
                    entry['expressions'].extend(expressions)
                    entry['dataset_sources'].append(dataset_name)
        
        print(f"Before merging: {sum(len(dataset[split]) for data_path in data_path_lst for dataset in [load_dataset(data_path)] for split in dataset.keys())} annotations")
        print(f"After merging: {len(merged_annotations)} unique objects")
        
        # Convert to final format
        final_data = []
        images_data = defaultdict(list)
        
        for dup_key, entry in merged_annotations.items():
            # Get category name from COCO mapping
            category_name = COCO_CATEGORIES.get(entry['category_id'], 'object')
            
            # Merge and enhance expressions
            merged_expressions = self.expression_merger.merge_expressions(entry['expressions'])
            
            annotation = {
                'category': category_name,
                'bbox': entry['bbox'],
                'descriptions': merged_expressions,
                'dataset_sources': list(set(entry['dataset_sources']))  # Remove duplicates
            }
            
            images_data[entry['image_path']].append(annotation)
        
        # Group by image and create final structure
        for image_path, annotations in images_data.items():
            anno_string = self.create_anno_string(annotations)
            
            data_entry = {
                "image_path": image_path,
                "anno": anno_string,
                "annotations": annotations,
                "questions": [],
                "responses": []
            }
            final_data.append(data_entry)
        
        # Print merge statistics
        total_expressions = sum(len(ann['descriptions']) for data in final_data for ann in data['annotations'])
        multi_dataset_objects = sum(1 for data in final_data for ann in data['annotations'] if len(ann['dataset_sources']) > 1)
        
        print(f"Final dataset statistics:")
        print(f"- Total images: {len(final_data)}")
        print(f"- Total objects: {sum(len(data['annotations']) for data in final_data)}")
        print(f"- Total expressions: {total_expressions}")
        print(f"- Objects from multiple datasets: {multi_dataset_objects}")
        
        return final_data
    
    def create_anno_string(self, annotations):
        """Create annotation string in required format"""
        anno_parts = []
        object_counter = defaultdict(int)
        
        for ann in annotations:
            category = ann['category']
            object_counter[category] += 1
            obj_num = object_counter[category]
            
            bbox = xywh_to_xyxy(ann['bbox'])
            bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
            
            descriptions = ', '.join([f'"{desc}"' for desc in ann['descriptions']])
            connector = "who is" if category == "person" else "which is"
            
            anno_parts.append(f"- {category} {obj_num} {bbox_str}, {connector} {descriptions}")
        
        return "\n".join(anno_parts)
    
    def generate_initial_questions(self, data_list):
        """Generate initial questions for each object"""
        print("Generating initial questions...")
        
        for data_entry in tqdm(data_list, desc="Processing images"):
            image_path = data_entry["image_path"]
            anno = data_entry["anno"]
            
            try:
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
            
            # Generate questions for each object
            object_counter = defaultdict(int)
            all_questions = []
            
            for ann in data_entry["annotations"]:
                category = ann['category']
                object_counter[category] += 1
                obj_num = object_counter[category]
                
                bbox = xywh_to_xyxy(ann['bbox'])
                bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
                
                question_prompt = f"""<image>
Image has these objects with bboxes and descriptions:
{anno}

Create two or three questions from the visuals of the {category} {obj_num} {bbox_str}."""
                
                try:
                    response = self.model.chat(self.tokenizer, pixel_values, question_prompt, generation_config)
                    questions = self.extract_questions_from_response(response)
                    all_questions.extend(questions)
                except Exception as e:
                    print(f"Error generating questions: {e}")
                    continue
            
            data_entry["questions"] = all_questions
        
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
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
            
            detailed_responses = []
            
            for question in data_entry["questions"]:
                if not question.strip():
                    continue
                
                try:
                    # Remove bbox from annotation string for this prompt, as per instructions
                    anno_for_prompt = re.sub(r' \[[^\]]*\]', '', anno)
                    # Initial detailed response
                    prompt_1 = self.question_2_comm.format(anno=anno_for_prompt, question=question)
                    response_1, history = self.model.chat(
                        self.tokenizer, pixel_values, prompt_1, generation_config, return_history=True
                    )
                    
                    # Follow-up refinement
                    response_2, _ = self.model.chat(
                        self.tokenizer, pixel_values, self.question_2_followup, generation_config, history=history
                    )
                    
                    detailed_responses.append({
                        "question": question,
                        "initial_response": response_1,
                        "final_response": response_2
                    })
                    
                except Exception as e:
                    print(f"Error in detailed response for '{question}': {e}")
                    continue
            
            data_entry["responses"] = detailed_responses
        
        return data_list
    
    def save_results(self, data_list, output_prefix="refcoco_merged_results"):
        """Save results with merge information"""
        print("Saving results...")
        
        # Enhanced JSON format with merge info
        json_output = []
        for entry in data_list:
            json_entry = {
                "image_path": entry["image_path"],
                "anno": entry["anno"],
                "merge_info": {
                    "total_objects": len(entry["annotations"]),
                    "multi_dataset_objects": len([ann for ann in entry["annotations"] if len(ann['dataset_sources']) > 1]),
                    "dataset_coverage": list(set(ds for ann in entry["annotations"] for ds in ann['dataset_sources']))
                },
                "questions_and_responses": entry["responses"]
            }
            json_output.append(json_entry)
        
        with open(f"{output_prefix}.json", "w", encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Python format
        with open(f"{output_prefix}.py", "w", encoding='utf-8') as f:
            f.write("# Enhanced RefCOCO dataset with merged referring expressions\n")
            f.write("# Generated from RefCOCO, RefCOCOplus, and RefCOCOg\n\n")
            f.write("data_refcoco_merged = [\n")
            
            for i, entry in enumerate(json_output):
                f.write("    {\n")
                f.write(f"        \"image_path\": \"{entry['image_path']}\",\n")
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
    """Main execution function"""
    processor = RefCOCOProcessor()
    
    # Step 1: Load and merge datasets
    data_list = processor.load_and_merge_datasets()
    
    # Step 2: Generate initial questions
    data_list = processor.generate_initial_questions(data_list)
    
    # Step 3: Generate detailed responses
    data_list = processor.generate_detailed_responses(data_list)
    
    # Step 4: Save results
    results = processor.save_results(data_list)
    
    # Print final statistics
    total_questions = sum(len(entry["questions_and_responses"]) for entry in results)
    multi_dataset_coverage = sum(entry["merge_info"]["multi_dataset_objects"] for entry in results)
    
    print(f"\nFinal Statistics:")
    print(f"Total images: {len(results)}")
    print(f"Total questions: {total_questions}")
    print(f"Average questions per image: {total_questions/len(results):.2f}")
    print(f"Objects enhanced by multi-dataset merging: {multi_dataset_coverage}")

if __name__ == "__main__":
    main()