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
from pathlib import Path

import torch

from InternVL3.utils.toolcall_parser_n_fixer import (
    extract_tool_calls,
    fix_tool_calling_strings,
    parse_annotations,
)
from InternVL3.utils.constants import generation_config
from InternVL3.utils.processor_utils import load_llm_model, load_models, split_model


class RefCOCOProcessor:
    def __init__(self, model_path="OpenGVLab/InternVL3-38B", output_path="/mnt/nas3/Data/coco"):
        if os.path.exists(output_path):
            self.dataset_p_root = str(Path(output_path).parent)
        elif os.path.exists(output_path.replace("nas3/Data", "nas1/data")):
            self.dataset_p_root = str(Path(output_path.replace("nas3/Data", "nas1/data")).parent)
        else:
            raise FileNotFoundError(f"Error! coco path not exists")
        self.output_folder = os.path.join(self.dataset_p_root, "coco", "refcoco_vlm_results_theo")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        print("Initializing model...")
        if model_path.split("/")[0] == "Qwen":
            self.model, self.tokenizer, self.sampling_params = load_llm_model(model_path)
        else:
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

        self.q3_prompt = """Refine the response below:

I will provide the original question and model's response. Original question includes object annotations, commands, and question at the end. The response includes thought process to reach to the answer to the question at the end of the original question. Model is meant to generate response only from visuals, not descriptions. Thus, existence of object annotations should not be mentioned in the response, and they should be rephrased. Response can include {{crop (object with number)}}, but be aware that crop (object with description) is invalid. Response should not mention object with number except for crop tool. Therefore, response should be fixed. Give me the fixed version of response.

# Original Question:
##Object names with descriptions
{prompt_2}

# Response:
{response_1}
"""
        self.q3_prompt_2 = """Refine the text below:
I will provide the original question and LLM's response. The original response includes thought process to reach to the answer to the original question. Model is meant to generate response only from visuals, not descriptions. Thus, existence of object annotations should not be mentioned in the response, and they should be rephrased. Response can include <T>, </T>, <A>, </A>, {{Tool call}}, and these tags should remain as is. Response should not include noun plus # pattern, like "person 1". Therefore, noun # pattern should be replaced with noun phrase with a participial modifier. Give me the fixed version of response.
# Original Question:
{question}

# Original Response:
{answer}

# Wrong patterns:
{wrong_pattern_str}
"""
        return

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

    def generate_initial_questions(self, data_entry, pixels):
        """Generate initial questions for each object in batch
        Return: the number of inferences."""
        anno = data_entry["annos_str"]
        all_qna = list()
        num_infer = 0

        for ann in anno.split("- "):
            if len(ann) < 1:
                continue

            target_obj = ann[: ann.find(" [") + 1].strip()
            prompt_1 = self.q1_prompt.format(ann=anno, target_obj=target_obj)
            with torch.inference_mode():
                response = self.model.chat(self.tokenizer, pixels, prompt_1, generation_config)
            num_infer += 1
            questions = self.parse_qna_from_response(response, lookfor="Question")
            answers = self.parse_qna_from_response(response, lookfor="Answer")
            qna1 = [
                {"Q": q1, "A1": a1.strip("}"), "A2": "", "A3": ""}
                for q1, a1 in zip(questions, answers)
            ]
            all_qna.extend(qna1)

        data_entry["QnA"] = all_qna
        return num_infer

    def generate_initial_questions_b(self, data_entry, pixels, batch_size=2):
        """Generate initial questions for each object in batch
        Return: the number of inferences & the number of total samples (<= inference x batch_size)."""
        anno = data_entry["annos_str"]
        all_qna = list()
        num_infer = 0
        total_samples = 0

        index = 0
        for i, ann in enumerate(anno.split("- ")):
            if len(ann) < 1:
                continue

            if index == 0:
                target_objs = [ann[: ann.find(" [") + 1].strip()]
                prompt_1s = [self.q1_prompt.format(ann=anno, target_obj=target_objs[-1])]
            elif index < batch_size:
                target_objs.append(ann[: ann.find(" [") + 1].strip())
                prompt_1s.append(self.q1_prompt.format(ann=anno, target_obj=target_objs[-1]))

            if index == batch_size - 1 or i == len(anno.split("- ")) - 1:
                num_patches_list = [pixels.size(0)] * len(prompt_1s)
                pixels = [pixels] * len(prompt_1s)
                pixels = torch.cat(pixels, dim=0)
                with torch.inference_mode():
                    responses = self.model.batch_chat(
                        self.tokenizer,
                        pixels,
                        num_patches_list=num_patches_list,
                        questions=prompt_1s,
                        generation_config=generation_config,
                    )
                num_infer += 1
                total_samples += len(responses)
                for response in responses:
                    questions = self.parse_qna_from_response(response, lookfor="Question")
                    answers = self.parse_qna_from_response(response, lookfor="Answer")
                    qna1 = [
                        {"Q": q1, "A1": a1.strip("}"), "A2": "", "A3": ""}
                        for q1, a1 in zip(questions, answers)
                    ]
                    all_qna.extend(qna1)

            index += 1
            if index == batch_size:
                index = 0

        data_entry["QnA"] = all_qna
        return num_infer, total_samples

    def parse_qna_from_response(self, response, lookfor="Question"):
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

    def generate_detailed_responses(self, data_entry, pixels):
        """Generate detailed responses with multimodal reasoning"""
        anno = data_entry["annos_str"]
        for qna in data_entry["QnA"]:
            question = qna["Q"]
            # Remove bbox from annotation string for this prompt
            anno_for_prompt = re.sub(r" \[[^\]]*\]", "", anno)
            prompt_2 = self.q2_prompt.format(anno=anno_for_prompt, question=question)
            with torch.inference_mode():
                response_2 = self.model.chat(self.tokenizer, pixels, prompt_2, generation_config)
            qna["A2"] = response_2.strip()
        return

    def generate_detailed_responses_b(self, data_entry, pixels, batch_size=2):
        """Generate detailed responses in batch with multimodal reasoning"""

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
                num_patches_list = [pixels.size(0)] * len(prompt_2s)
                pixels = [pixels] * len(prompt_2s)
                pixels = torch.cat(pixels, dim=0)
                with torch.inference_mode():
                    responses = self.model.batch_chat(
                        self.tokenizer,
                        pixels,
                        num_patches_list=num_patches_list,
                        questions=prompt_2s,
                        generation_config=generation_config,
                    )
                answers.extend(responses)
            index += 1
            if index == batch_size:
                index = 0

        for qna, response_2 in zip(data_entry["QnA"], answers):
            qna["A2"] = response_2.strip()

        return

    def fix_answer_strings(self, data_entry):
        for qna in data_entry["QnA"]:
            if "A3" in qna and len(qna["A3"]) > 100:
                continue
            response = fix_tool_calling_strings(
                data_entry["image_id"], qna["A2"], data_entry["annos_str"]
            )
            # Remove adundant think tags
            first_think_pos = response.find("<think>")
            last_think_pos = response.rfind("</think>")
            response = (
                response[: first_think_pos + len("<think>")]
                + response[first_think_pos + len("<think>") : last_think_pos]
                .replace("<think>", "")
                .replace("</think>", "")
                + response[last_think_pos:]
            )
            annotations = parse_annotations(data_entry["annos_str"])
            tool_calls = extract_tool_calls(response)
            tool_call_contents = [
                response[start_pos:end_pos] for _, start_pos, end_pos in tool_calls
            ]
            text_only = re.sub(r"\{.*?\}", "{Tool call}", response)
            wrong_pattern = [_ for _ in annotations.keys() if _ in text_only]

            if len(wrong_pattern) == 0:
                qna["A3"] = response
                continue

            text_only = (
                text_only.replace("<think>", "<T>")
                .replace("</think>", "</T>")
                .replace("<answer>", "<A>")
                .replace("</answer>", "</A>")
            )

            wrong_pattern_str = "\n".join(
                [
                    f"{_} should be replaced by rephrasing the annotations {annotations[_]}"
                    for _ in wrong_pattern
                ]
            )
            prompt = self.q3_prompt_2.format(
                question=qna["Q"], answer=text_only, wrong_pattern_str=wrong_pattern_str
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that refines multimodal reasoning responses.",
                },
                {"role": "user", "content": prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            with torch.inference_mode():
                outputs = self.model.generate(prompt, self.sampling_params)
            res = outputs[0].outputs[0].text  # TODO: must post-process to find answer only
            res = res[res.find("</think>") + len("</think>") :].strip()
            res = res[:3] + "\n" + res[3:-4].strip() + "\n" + res[-4:]
            res = (
                res.replace("<T>", "<think>")
                .replace("</T>", "</think>")
                .replace("<A>", "<answer>")
                .replace("</A>", "</answer>")
            )

            for toll_call_content in tool_call_contents:
                # replace {Tool call} with toll_call_content one by one
                loc = res.find("{Tool call}")
                if loc == -1:
                    print("Error! missing {Tool call} in the response")
                    qna["A3"] = ""
                    break
                res = (
                    res[:loc].strip()
                    + "\n"
                    + toll_call_content
                    + "\n"
                    + res[loc + len("{Tool call}") :].strip()
                )
            res = "\n".join([_.strip(" ") for _ in res.split("\n")])
            qna["A3"] = res
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
