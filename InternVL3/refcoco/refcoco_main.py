import json
import torch
from datasets import load_dataset
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import re

# Your InternVL3 imports
from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor import load_models, split_model

def xywh_to_xyxy(box):
    bbox = deepcopy(box)
    bbox[2] = box[0] + box[2]
    bbox[3] = box[1] + box[3]
    return bbox

class RefCOCOProcessor:
    def __init__(self, model_path="OpenGVLab/InternVL3-78B"):
        self.dataset_root = "/mnt/nas-1/data/coco"
        print("Initializing model...")
        device_map = split_model(model_path)
        self.model, self.tokenizer = load_models(model_path, device_map)
        
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
    
    def load_datasets(self):
        """Load and process RefCOCO datasets"""
        data_path_lst = [
            "jxu124/RefCOCO",
            "jxu124/RefCOCOplus",
            "jxu124/RefCOCOg",
        ]
        
        all_data = []
        
        for data_path in data_path_lst:
            print(f"Loading dataset: {data_path}")
            dataset = load_dataset(data_path)
            
            # Group by image
            images_data = defaultdict(list)
            
            for split in dataset.keys():
                for sample in tqdm(dataset[split], desc=f"Processing {data_path}/{split}"):
                    image_path = sample["image_path"]
                    
                    # Extract category from the sample or infer from context
                    category = sample.get('category', 'object')
                    if not category and 'person' in ' '.join(sample['answer']).lower():
                        category = 'person'
                    
                    annotation = {
                        'category': category,
                        'bbox': sample['bbox'],
                        'descriptions': sample['captions'] if isinstance(sample['captions'], list) else [sample['captions']]
                    }
                    
                    images_data[image_path].append(annotation)
            
            # Convert to required format
            for image_path, annotations in images_data.items():
                anno_string = self.create_anno_string(annotations)
                
                data_entry = {
                    "image_path": image_path,
                    "anno": anno_string,
                    "annotations": annotations,
                    "questions": [],
                    "responses": []
                }
                all_data.append(data_entry)
        
        return all_data
    
    def create_anno_string(self, annotations):
        """Create annotation string in required format"""
        anno_parts = []
        object_counter = defaultdict(int)
        
        for ann in annotations:
            category = ann['category']
            object_counter[category] += 1
            obj_num = object_counter[category]
            
            bbox = xywh_to_xyxy(ann['bbox'])
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            
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
                bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                
                question_prompt = f"""<image>
Image has these objects with bboxes and descriptions:
{anno}

Create two or three questions from the visuals of the {category} {obj_num} {bbox_str}. Generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly write the descriptions in your response."""
                
                try:
                    response = self.model.chat(self.tokenizer, pixel_values, question_prompt, generation_config)
                    questions = self.extract_questions_from_response(response)
                    all_questions.extend(questions)
                except Exception as e:
                    print(f"Error generating questions: {e}")
                    continue
            
            data_entry["questions"] = all_questions
        
        return data_list
    
    def extract_questions_from_response(self, response):
        """Extract individual questions from model response"""
        questions = []
        
        # Look for numbered questions
        pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
        
        if matches:
            questions = [match.strip() for match in matches]
        else:
            # Fallback: split by lines
            lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('#')]
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
                    # Initial detailed response
                    prompt_1 = self.question_2_comm.format(anno=anno, question=question)
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
    
    def save_results(self, data_list, output_prefix="refcoco_results"):
        """Save results in multiple formats"""
        print("Saving results...")
        
        # JSON format
        json_output = []
        for entry in data_list:
            json_entry = {
                "image_path": entry["image_path"],
                "anno": entry["anno"],
                "questions_and_responses": entry["responses"]
            }
            json_output.append(json_entry)
        
        with open(f"{output_prefix}.json", "w", encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        # Python format (similar to original questions.py)
        with open(f"{output_prefix}.py", "w", encoding='utf-8') as f:
            f.write("data_refcoco_generated = [\n")
            
            for i, entry in enumerate(json_output):
                f.write("    {\n")
                f.write(f"        \"image_path\": \"{entry['image_path']}\",\n")
                f.write(f"        \"anno\": \"\"\"\n{entry['anno']}\n\"\"\",\n")
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
    
    # Step 1: Load datasets
    data_list = processor.load_datasets()
    print(f"Loaded {len(data_list)} images")
    
    # Step 2: Generate initial questions
    data_list = processor.generate_initial_questions(data_list)
    
    # Step 3: Generate detailed responses
    data_list = processor.generate_detailed_responses(data_list)
    
    # Step 4: Save results
    results = processor.save_results(data_list)
    
    # Print statistics
    total_questions = sum(len(entry["questions_and_responses"]) for entry in results)
    print(f"\nFinal Statistics:")
    print(f"Total images: {len(results)}")
    print(f"Total questions: {total_questions}")
    print(f"Average questions per image: {total_questions/len(results):.2f}")

if __name__ == "__main__":
    main()