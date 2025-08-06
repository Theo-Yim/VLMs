"""
RefCOCO Dataset Verification and Fix Missing A2 Responses

This script loads JSON files created by refcoco_main3.py and checks for missing A2 responses.
When A2 is missing or empty, it generates the response using the model and saves the updated JSON.
"""

import json
import os
import re
from pathlib import Path
from tqdm import tqdm
import torch

from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_utils import load_models, split_model


class RefCOCOVerifier:
    def __init__(self, model_path="OpenGVLab/InternVL3-38B", output_path="/mnt/nas3/Data/coco"):
        if os.path.exists(output_path):
            self.dataset_p_root = str(Path(output_path).parent)
        elif os.path.exists(output_path.replace("nas3/Data", "nas1/data")):
            self.dataset_p_root = str(Path(output_path.replace("nas3/Data", "nas1/data")).parent)
        else:
            raise FileNotFoundError(f"Error! coco path not exists")
        
        self.output_folder = os.path.join(self.dataset_p_root, "coco", "refcoco_vlm_results_theo")
        if not os.path.exists(self.output_folder):
            raise FileNotFoundError(f"Output folder not found: {self.output_folder}")

        print("Initializing model...")
        device_map = split_model(model_path)
        self.model, self.tokenizer = load_models(model_path, device_map)

        self.q2_prompt = """<image>
##Object names with descriptions
{anno}
Since there are many objects, you should use them enough to create a well-aligned response.
##
You are performing "Multimodal Interleaved Reasoning". During the thinking process, keep an eye on the visual cues in the original image, identify regions that help answer the question, and use the "Crop" tool to crop and zoom in for detailed analysis.
When using the tool, you must output a JSON object in the following format:
{{Crop (object name w number)}}
Ensure that you "Crop" at least once. You can simultaneously crop multiple adjacent objects to inspect them sufficiently. If you crop the region including multiple objects, list them with commas. For example, {{Crop person 1, desk 3}}, {{Crop person 2}}.
Continue thinking after each operation until you reach the final answer. Output the thinking process within a pair of <think> </think> tags and then output the final answer within a pair of <answer> </answer> tags. Do not use an object name with a number outside the crop tool, but use a noun phrase with a description instead.
Question: {question}"""

    def find_json_files(self, directory=None):
        """Find all JSON files in the output directory"""
        if directory is None:
            directory = self.output_folder
        
        json_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        return json_files

    def load_json_file(self, json_path):
        """Load and validate JSON file structure"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ['image_path', 'image_id', 'annos_str', 'QnA']
            for field in required_fields:
                if field not in data:
                    print(f"Warning: Missing required field '{field}' in {json_path}")
                    return None
            
            return data
        
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return None

    def check_missing_a2(self, qna_list):
        """Check for missing A2 responses in QnA list"""
        missing_indices = []
        
        for i, qna in enumerate(qna_list):
            if 'A2' not in qna or not qna['A2'] or qna['A2'].strip() == "" or len(qna['A2'].strip()) < 100:
                missing_indices.append(i)
        
        return missing_indices

    def generate_detailed_responses_fix(self, json_path, save_updated=True):
        """
        Generate missing A2 responses for a single JSON file
        
        Args:
            json_path: Path to the JSON file
            save_updated: Whether to save the updated JSON file
            
        Returns:
            tuple: (total_qna, missing_count, fixed_count)
        """
        print(f"Processing: {json_path}")
        
        # Load JSON data
        data = self.load_json_file(json_path)
        if data is None:
            return 0, 0, 0
        
        # Check for missing A2 responses
        missing_indices = self.check_missing_a2(data['QnA'])
        total_qna = len(data['QnA'])
        missing_count = len(missing_indices)
        
        if missing_count == 0:
            print(f"  ✓ No missing A2 responses found")
            return total_qna, missing_count, 0
        
        print(f"  Found {missing_count}/{total_qna} missing A2 responses")
        
        # Load image
        image_path = os.path.join(self.dataset_p_root, data['image_path'])
        try:
            pixel_values = (
                load_image(image_path, max_num=12)
                .to(torch.bfloat16)
                .cuda()
            )
        except Exception as e:
            print(f"  Error loading image {image_path}: {e}")
            return total_qna, missing_count, 0
        
        # Prepare annotation string without bboxes for q2_prompt
        anno_for_prompt = re.sub(r" \[[^\]]*\]", "", data['annos_str'])
        
        # Generate missing A2 responses
        fixed_count = 0
        for idx in missing_indices:
            qna = data['QnA'][idx]
            question = qna['Q']
            
            try:
                prompt_2 = self.q2_prompt.format(anno=anno_for_prompt, question=question)
                response_2 = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    prompt_2, 
                    generation_config
                )
                
                # Update the A2 response
                qna['A2'] = response_2.strip()
                fixed_count += 1
                print(f"    ✓ Fixed A2 for question {idx + 1}: {question[:50]}...")
                
            except Exception as e:
                print(f"    ✗ Error generating A2 for question {idx + 1}: {e}")
                continue
        
        # Save updated JSON if requested
        if save_updated and fixed_count > 0:
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  ✓ Saved updated JSON with {fixed_count} fixes")
            except Exception as e:
                print(f"  ✗ Error saving updated JSON: {e}")
        
        return total_qna, missing_count, fixed_count

    def batch_fix_missing_responses(self, directory=None, pattern="*.json"):
        """
        Fix missing A2 responses for all JSON files in a directory
        
        Args:
            directory: Directory to search for JSON files (default: output_folder)
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            dict: Summary statistics
        """
        if directory is None:
            directory = self.output_folder
        
        # Find all JSON files
        json_files = self.find_json_files(directory)
        
        if not json_files:
            print(f"No JSON files found in {directory}")
            return {}
        
        print(f"Found {len(json_files)} JSON files to process")
        
        # Process statistics
        total_files = len(json_files)
        processed_files = 0
        total_qna_count = 0
        total_missing_count = 0
        total_fixed_count = 0
        
        # Process each file
        for json_path in tqdm(json_files, desc="Processing JSON files"):
            qna_count, missing_count, fixed_count = self.generate_detailed_responses_fix(json_path)
            
            if qna_count > 0:  # Only count successfully processed files
                processed_files += 1
                total_qna_count += qna_count
                total_missing_count += missing_count
                total_fixed_count += fixed_count
        
        # Print summary
        print(f"\n=== Processing Summary ===")
        print(f"Files processed: {processed_files}/{total_files}")
        print(f"Total QnA entries: {total_qna_count}")
        print(f"Missing A2 responses: {total_missing_count}")
        print(f"Fixed A2 responses: {total_fixed_count}")
        if total_missing_count > 0:
            print(f"Fix rate: {total_fixed_count/total_missing_count*100:.1f}%")
        
        return {
            'total_files': total_files,
            'processed_files': processed_files,
            'total_qna_count': total_qna_count,
            'total_missing_count': total_missing_count,
            'total_fixed_count': total_fixed_count,
            'fix_rate': total_fixed_count/total_missing_count*100 if total_missing_count > 0 else 0
        }

    def verify_single_file(self, json_path, fix_if_missing=True):
        """
        Verify and optionally fix a single JSON file
        
        Args:
            json_path: Path to the JSON file
            fix_if_missing: Whether to fix missing A2 responses
            
        Returns:
            dict: File statistics
        """
        data = self.load_json_file(json_path)
        if data is None:
            return {'error': 'Failed to load JSON file'}
        
        missing_indices = self.check_missing_a2(data['QnA'])
        
        result = {
            'file_path': json_path,
            'total_qna': len(data['QnA']),
            'missing_a2': len(missing_indices),
            'missing_indices': missing_indices,
            'needs_fix': len(missing_indices) > 0
        }
        
        if fix_if_missing and len(missing_indices) > 0:
            _, _, fixed_count = self.generate_detailed_responses_fix(json_path)
            result['fixed_count'] = fixed_count
        
        return result


def main():
    """
    Main execution function for verifying and fixing missing A2 responses
    """
    verifier = RefCOCOVerifier(model_path="OpenGVLab/InternVL3-38B")
    
    # Option 1: Fix all JSON files in the output directory
    print("=== Batch Processing All JSON Files ===")
    stats = verifier.batch_fix_missing_responses()
    
    # Option 2: Verify a specific file (uncomment to use)
    # specific_file = "/path/to/specific/file.json"
    # result = verifier.verify_single_file(specific_file, fix_if_missing=True)
    # print(f"Single file result: {result}")
    
    print("=== Verification Complete ===")


if __name__ == "__main__":
    main()