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
from InternVL3.utils.processor import load_models, split_model


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

    def generate_initial_questions(self, data_entry, pixel_values, batch_size=2):
        """Generate initial questions for each object"""

        anno = data_entry["annos_str"]

        # Generate questions for each object
        all_qna = list()

        index = 0

        for i, ann in enumerate(anno.split("- ")):
            if len(ann) < 1:
                continue

            if index == 0:
                target_objs = [ann[: ann.find(" [") + 1].strip()]
                # Enhanced question prompt that mentions the quality of merged referring expressions
                prompt_1s = [self.q1_prompt.format(ann=anno, target_obj=target_objs[-1])]
            elif index < batch_size:
                target_objs.append(ann[: ann.find(" [") + 1].strip())
                # Enhanced question prompt that mentions the quality of merged referring expressions
                prompt_1s.append(self.q1_prompt.format(ann=anno, target_obj=target_objs[-1]))

            if index == batch_size - 1 or i == len(anno.split("- ")) - 1:
                num_patches_list = [pixel_values.size(0)] * len(prompt_1s)
                pixel_values = [pixel_values] * len(prompt_1s)
                pixel_values = torch.cat(pixel_values, dim=0)
                responses = self.model.batch_chat(self.tokenizer, pixel_values, 
                                           num_patches_list=num_patches_list, questions=prompt_1s,
                                           generation_config=generation_config)

                for response in responses:
                    questions = self.extract_questions_n_answers_from_response(
                        response, lookfor="Question"
                    )
                    answers = self.extract_questions_n_answers_from_response(response, lookfor="Answer")
                    qna1 = [
                        {"Q": q1, "A1": a1.strip("}"), "A2": "", "A3": ""}
                        for q1, a1 in zip(questions, answers)
                    ]
                    all_qna.extend(qna1)

            index += 1
            if index == batch_size:
                index = 0

        data_entry["QnA"] = all_qna
        # data_entry["responses_1"] = all_answers

        return

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

    def generate_detailed_responses(self, data_entry, pixel_values, batch_size=2):
        """Generate detailed responses with multimodal reasoning"""

        anno = data_entry["annos_str"]
        # Remove bbox from annotation string for this prompt
        anno_for_prompt = re.sub(r" \[[^\]]*\]", "", anno)

        index = 0
        answers = list()

        for i, qna in enumerate(data_entry["QnA"]):
            if index == 0:
                question = qna["Q"]
                # Initial detailed response
                prompt_2s = [self.q2_prompt.format(anno=anno_for_prompt, question=question)]
            elif index < batch_size:
                question = qna["Q"]
                prompt_2s.append(self.q2_prompt.format(anno=anno_for_prompt, question=question))

            if index == batch_size - 1 or i == len(data_entry["QnA"]) - 1:
                num_patches_list = [pixel_values.size(0)] * len(prompt_2s)
                pixel_values = [pixel_values] * len(prompt_2s)
                pixel_values = torch.cat(pixel_values, dim=0)
                responses = self.model.batch_chat(
                    self.tokenizer, pixel_values, num_patches_list=num_patches_list,
                    questions=prompt_2s, generation_config=generation_config
                )
                answers.extend(responses)
            index += 1
            if index == batch_size:
                index = 0
        
        for qna, response_2 in zip(data_entry["QnA"], answers):
            qna["A2"] = response_2.strip()

        return

    def save_results(self, data_entry, output_path):
        """Save results with merge statistics"""

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, indent=2, ensure_ascii=False)

        # # Python format
        # with open(output_path + ".py", "w", encoding="utf-8") as f:
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

        # print(f"Results saved to {data_entry["image_id"]}.json")
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

    for data_entry in tqdm(data_list, desc="Processing images"):
        image_path = data_entry["image_path"]
        output_path = os.path.join(processor.output_folder, data_entry["image_id"] + ".json")
        if os.path.exists(output_path):
            print(f"Skipping {image_path}, already processed.")
            continue
        try:
            pixel_values = (
                load_image(os.path.join(processor.dataset_p_root, image_path), max_num=12)
                .to(torch.bfloat16)
                .cuda()
            )
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
        processor.generate_initial_questions(data_entry, pixel_values)
        processor.generate_detailed_responses(data_entry, pixel_values)
        # Save results
        # processor.save_results(data_entry, output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, indent=2, ensure_ascii=False)
        # print(f"Processed {image_path}.")


if __name__ == "__main__":
    main()
