"""
RefCOCO Dataset Processing with Intelligent Referring Expression Merging

Key Insight: RefCOCO, RefCOCOplus, and RefCOCOg all use the same underlying COCO annotations,
so bounding boxes are identical. The real value is in merging diverse referring expressions
from different datasets to create richer, more comprehensive descriptions.

This approach:
1. Uses proper COCO category mapping via category_id
2. Loads pre-merged referring expressions from different datasets
3. Generates enhanced questions and multimodal reasoning responses

Note: Run merge_refcoco_datasets.py first to create the merged dataset file.

This version:
- processes entire images step-by-step and saves the entire results in a single JSON file
"""

import json
import os
import pickle
import re

import torch
from tqdm import tqdm

# Your InternVL3 imports
from InternVL3.utils.constants import generation_config
from InternVL3.utils.preprocess import load_image
from InternVL3.utils.processor_utils import load_models, split_model


class RefCOCOProcessor:
    def __init__(self, model_path="OpenGVLab/InternVL3-38B"):
        if os.path.exists("/mnt/nas3/Data/coco"):
            self.dataset_p_root = "/mnt/nas3/Data/"
        elif os.path.exists("/mnt/data/coco"):
            self.dataset_p_root = "/mnt/data/"
        else:
            raise FileNotFoundError(f"Error! coco path not exists")
        self.output_folder = os.path.join(self.dataset_p_root, "coco", "refcoco_vlm_results_theo")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        print("Initializing model...")
        device_map = split_model(model_path)
        self.model, self.tokenizer = load_models(model_path, device_map)

        self.q1_prompt = """<image>
Image has these objects with bboxes and descriptions:
{ann}
Create two questions from the visuals of the {target_obj}. Generate a detailed reasoning from the visual clue or visual relationships, focusing on the region, and based on reasoning, give me final answer. Do not explicitly mention numberred object name or the presence of the descriptions in your response.
Output format: {{Question: ...\nReasoning: ...\nAnswer: ...}}"""

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

        self.question_2_followup = "Check your previous answer, if it has logical error, or misuse of Crop tool, or wrong format. After you fix those, give me final answer"

    def load_datasets(self, merged_data_file="merged_refcoco_data.pkl"):
        """Load pre-merged RefCOCO datasets"""
        print(f"Loading pre-merged dataset from {merged_data_file}...")

        if not os.path.exists(merged_data_file):
            raise FileNotFoundError(
                f"Merged dataset file '{merged_data_file}' not found. "
                "Please run merge_refcoco_datasets.py first to create the merged dataset."
            )

        with open(merged_data_file, "rb") as f:
            merged_data = pickle.load(f)

        data_list = merged_data["data"]
        merge_stats = merged_data["merge_statistics"]

        print(f"Loaded merged dataset:")
        print(f"- Total images: {merge_stats['total_images']}")
        print(f"- Total objects: {merge_stats['total_annotations']}")
        print(
            f"- Objects with merged referring expressions: {merge_stats['annotations_with_merged_expressions']}"
        )
        print(f"- Merge rate: {merge_stats['merge_rate']:.1f}%")

        return data_list

    def generate_initial_questions(self, data_list):
        """Generate initial questions for each object"""
        print("Generating initial questions...")

        for data_entry in tqdm(data_list, desc="Processing images"):
            image_path = data_entry["image_path"]
            anno = data_entry["annos_str"]

            try:
                pixel_values = (
                    load_image(os.path.join(self.dataset_p_root, image_path), max_num=12)
                    .to(torch.bfloat16)
                    .cuda()
                )
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

            # Generate questions for each object
            all_qna = list()

            for ann in anno.split("- "):
                if len(ann) < 1:
                    continue

                target_obj = ann[: ann.find(" [") + 1].strip()
                # Enhanced question prompt that mentions the quality of merged referring expressions
                prompt_1 = self.q1_prompt.format(ann=anno, target_obj=target_obj)

                try:
                    response = self.model.chat(
                        self.tokenizer, pixel_values, prompt_1, generation_config
                    )
                    questions = self.extract_questions_n_answers_from_response(
                        response, lookfor="Question"
                    )
                    answers = self.extract_questions_n_answers_from_response(
                        response, lookfor="Answer"
                    )
                    qna1 = [
                        {"Q": q1, "A1": a1.strip("}"), "A2": "", "A3": ""}
                        for q1, a1 in zip(questions, answers)
                    ]
                    all_qna.extend(qna1)
                    # all_questions.extend(questions)
                    # all_answers.extend(answers)
                except Exception as e:
                    print(f"Error generating questions: {e}")
                    continue

            data_entry["QnA"] = all_qna
            # data_entry["responses_1"] = all_answers

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
            fb1_questions = [
                line.strip()[line.find(":") + 1 :].strip()
                for line in response.split("\n")
                if lookfor.lower() in line.lower()
            ]
            questions = [q for q in fb1_questions if q]

        if not questions:
            # Fallback 2: Look for numbered list questions (e.g., "1. What is...")
            pattern = r"\d+\.\s*(.+?)(?=\d+\.|$)"
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                questions = [match.strip() for match in matches]

        if not questions:
            # Fallback 3: Look for any line with a question mark.
            lines = [
                line.strip()
                for line in response.split("\n")
                if line.strip() and not line.strip().startswith("#") and "?" in line
            ]
            questions = lines[:3]  # Limit to 3 questions

        return questions

    def generate_detailed_responses(self, data_list):
        """Generate detailed responses with multimodal reasoning"""
        print("Generating detailed responses...")

        for data_entry in tqdm(data_list, desc="Detailed responses"):
            image_path = data_entry["image_path"]
            anno = data_entry["annos_str"]

            try:
                pixel_values = (
                    load_image(os.path.join(self.dataset_p_root, image_path), max_num=12)
                    .to(torch.bfloat16)
                    .cuda()
                )
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue

            responses_2 = []

            for qna in data_entry["QnA"]:
                try:
                    question = qna["Q"]
                    # Remove bbox from annotation string for this prompt
                    anno_for_prompt = re.sub(r" \[[^\]]*\]", "", anno)
                    # Initial detailed response
                    prompt_2 = self.q2_prompt.format(anno=anno_for_prompt, question=question)

                except Exception as e:
                    print(f"Error formatting '{anno}': {e}")
                    continue
                response_2 = self.model.chat(
                    self.tokenizer, pixel_values, prompt_2, generation_config
                )  # , return_history=True)

                # # Follow-up refinement
                # response_3 = self.model.chat(self.tokenizer, pixel_values, self.question_2_followup, generation_config, history=history)

                responses_2.append(response_2)
                qna["A2"] = response_2.strip()

            # data_entry["responses_2"] = responses_2

        return data_list

    def save_results(self, data_list, output_prefix="refcoco_merged_results"):
        """Save results with merge statistics"""
        print("Saving results...")

        # Calculate merge statistics
        total_merged = sum(
            1
            for entry in data_list
            for ann in entry["annotations"]
            if ann.get("merged_from", 1) > 1
        )
        total_annotations = sum(len(entry["annotations"]) for entry in data_list)

        print(f"Merge Statistics:")
        print(f"Total objects: {total_annotations}")
        print(f"Objects with merged referring expressions: {total_merged}")
        print(f"Merge rate: {total_merged / total_annotations * 100:.1f}%")

        # JSON format with merge info
        json_output = []
        for entry in data_list:
            json_entry = {
                "image_path": entry["image_path"],
                "image_id": entry["image_id"],
                "annos_str": entry["annos_str"],
                "merge_info": {
                    "total_objects": len(entry["annotations"]),
                    "objects_with_merged_expressions": sum(
                        1 for ann in entry["annotations"] if ann.get("merged_from", 1) > 1
                    ),
                    "dataset_sources": list(
                        set(
                            source
                            for ann in entry["annotations"]
                            for source in ann.get("dataset_sources", [])
                        )
                    ),
                },
                "QnA": entry["QnA"],
            }
            json_output.append(json_entry)

        with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        # # Python format
        # with open(f"{output_prefix}.py", "w", encoding="utf-8") as f:
        #     f.write("# RefCOCO dataset with intelligent merging of referring expressions\n")
        #     f.write(
        #         f"# Merge statistics: {total_merged}/{total_annotations} objects have merged referring expressions\n\n"
        #     )
        #     f.write("data_refcoco_merged = [\n")

        #     for i, entry in enumerate(json_output):
        #         f.write("    {\n")
        #         f.write(f'        "image_path": "{entry["image_path"]}",\n')
        #         f.write(f'        "image_id": "{entry["image_id"]}",\n')
        #         f.write(f'        "anno": """\n{entry["annos_str"]}\n""",\n')
        #         f.write('        "questions": [')

        #         questions = [qr["Q"] for qr in entry["QnA"]]
        #         f.write(", ".join([f'"{q}"' for q in questions]))
        #         f.write("],\n")

        #         f.write('        "responses": [\n')
        #         for qr in entry["QnA"]:
        #             f.write(f'            """{qr["A2"]}""",\n')
        #         f.write("        ]\n")

        #         f.write("    },\n" if i < len(json_output) - 1 else "    }\n")

        #     f.write("]\n")

        # print(f"Results saved to {output_prefix}.json and {output_prefix}.py")
        return


def main():
    """
    Main execution function

    Key insight: RefCOCO datasets share the same COCO annotations (identical bboxes),
    but have different referring expressions. We merge these expressions rather than bboxes.

    Note: Run merge_refcoco_datasets.py first to create the merged dataset file.
    """
    processor = RefCOCOProcessor(model_path="OpenGVLab/InternVL3-38B")

    # Load pre-merged datasets
    data_list = processor.load_datasets()
    print(f"Loaded {len(data_list)} unique images with merged referring expressions")

    # data_list = data_list[:360]  # For testing, limit to first 2 images

    # Generate initial questions
    data_list = processor.generate_initial_questions(data_list)

    # Generate detailed responses
    data_list = processor.generate_detailed_responses(data_list)

    # Generate object_localization questions and responses

    # Save results
    processor.save_results(data_list)

    # # Print final statistics
    # total_questions = sum(len(entry["QnA"]) for entry in results)
    # print(f"\nFinal Statistics:")
    # print(f"Unique images: {len(results)}")
    # print(f"Total questions: {total_questions}")
    # print(f"Average questions per image: {total_questions / len(results):.2f}")


if __name__ == "__main__":
    main()
