"""
Stage 2 Multi: Refine and Convert Multi-Person Identity Dataset

Processes Stage 1 Multi outputs:
1. Fixes malformed questions and answers
2. Ensures proper <think> and <answer> structure with MULTIPLE tool calls
3. Replaces {{Identify person #X}} with actual tool calls (multiple per answer)
4. Enriches answers to include all identified people
5. Converts to final conversation format with system prompt
6. Tracks metrics
"""

import json
import os
import re
from typing import Dict, List, Tuple


class IdentityStage2MultiProcessor:
    """Stage 2 Multi: Refine and convert multi-person Q&A to final format"""

    # System prompt (same as single-person)
    SYSTEM_PROMPT = """You have access to an Identify tool for verifying people's identities in images. When you need to identify a person, use the tool in this format: <tool_call>Identify [x, y, x2, y2]</tool_call>

After each tool use, the person's identity will be provided as: <tool_response>Name</tool_response>. Use this capability to provide accurate identity-based responses."""

    def __init__(self, stage1_multi_folder: str, output_file: str):
        self.stage1_multi_folder = stage1_multi_folder
        self.output_file = output_file

        # Metrics
        self.metrics = {
            "total_images": 0,
            "total_qna": 0,
            "malformed_questions": 0,
            "fixed_tool_calls": 0,
            "missing_think_tags": 0,
            "missing_answer_tags": 0,
            "enriched_answers": 0,
            "removed_qna": 0,
            "multi_tool_calls": 0,  # Track questions with 2+ tool calls
            "total_tool_calls": 0,
        }

    def load_stage1_multi_files(self) -> List[Dict]:
        """Load all Stage 1 Multi JSON files"""
        json_files = []
        for file in os.listdir(self.stage1_multi_folder):
            if file.endswith(".json"):
                filepath = os.path.join(self.stage1_multi_folder, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    json_files.append(data)

        print(f"Loaded {len(json_files)} Stage 1 Multi files")
        return json_files

    def fix_question(self, question: str) -> Tuple[str, bool]:
        """
        Fix malformed questions
        Returns: (fixed_question, was_malformed)
        """
        original = question
        is_malformed = False

        # Handle multiline or incomplete questions
        if "\n" in question or not question.strip().endswith("?"):
            is_malformed = True
            lines = question.split("\n")
            candidates = []

            for line in lines:
                line = line.strip()
                line = re.sub(r'^["\']|["\']$', "", line)
                if line.endswith("?") and len(line.split()) >= 6:
                    candidates.append(line)

            if candidates:
                question = max(candidates, key=len)
            else:
                for line in lines:
                    line = line.strip()
                    if any(
                        qword in line.lower()
                        for qword in ["who is", "who are", "what is", "can you", "which"]
                    ):
                        question = line
                        if not question.endswith("?"):
                            question += "?"
                        break
                else:
                    return None, True

        # Remove quotes and extra whitespace
        question = re.sub(r'^["\']|["\']$', "", question.strip())
        question = " ".join(question.split())

        # Ensure it starts with capital
        if question and question[0].islower():
            question = question[0].upper() + question[1:]

        # Ensure it ends with ?
        if not question.endswith("?"):
            question += "?"
            is_malformed = True

        return question, (is_malformed or original != question)

    def enrich_multi_person_answer(
        self, answer_content: str, question: str, person_names: List[str]
    ) -> str:
        """
        Enrich multi-person answer to include all identified people

        Examples:
        - "The people are X and Y." -> "The people visible in the image are X (left) and Y (right)."
        - "They are A, B, C." -> "The three people are A, B, and C."
        """
        # If answer already includes good detail (>15 words), keep it
        if len(answer_content.split()) > 15:
            return answer_content

        # Build enriched answer based on number of people
        num_people = len(person_names)

        if "who are all" in question.lower() or "everyone" in question.lower():
            # Group identification
            if num_people == 2:
                enriched = f"The people in the image are {person_names[0]} and {person_names[1]}."
            elif num_people == 3:
                enriched = f"The three people visible are {person_names[0]}, {person_names[1]}, and {person_names[2]}."
            else:
                names_str = ", ".join(person_names[:-1]) + f", and {person_names[-1]}"
                enriched = f"The {num_people} people in the image are {names_str}."

        elif "two people" in question.lower():
            # Two-person selective
            enriched = f"The two people are {person_names[0]} and {person_names[1]}."

        elif "left to right" in question.lower() or "from left" in question.lower():
            # Sequential
            names_str = ", ".join(person_names[:-1]) + f", and {person_names[-1]}"
            enriched = f"From left to right, the people are {names_str}."

        else:
            # Generic fallback
            if num_people == 2:
                enriched = f"The two individuals are {person_names[0]} and {person_names[1]}."
            else:
                names_str = ", ".join(person_names[:-1]) + f", and {person_names[-1]}"
                enriched = f"The people are {names_str}."

        self.metrics["enriched_answers"] += 1
        return enriched

    def fix_answer(
        self, answer_raw: str, question: str, person_names: List[str]
    ) -> Tuple[str, Dict]:
        """
        Fix multi-person answer formatting
        Handles MULTIPLE tool call placeholders
        Returns: (fixed_answer, fix_stats)
        """
        stats = {
            "missing_think": False,
            "missing_answer": False,
        }

        # Ensure <think> and <answer> tags exist
        has_think = "<think>" in answer_raw and "</think>" in answer_raw
        has_answer = "<answer>" in answer_raw and "</answer>" in answer_raw

        if not has_think:
            stats["missing_think"] = True
        if not has_answer:
            stats["missing_answer"] = True

        # If both missing, reconstruct
        if not has_think and not has_answer:
            # Extract all placeholders
            placeholders = re.findall(r"\{\{Identify person #\d+\}\}", answer_raw)
            if placeholders:
                think = answer_raw.strip()
                # Create enriched answer
                enriched = self.enrich_multi_person_answer("", question, person_names)
                answer_raw = f"<think>\n{think}\n</think>\n<answer>{enriched}</answer>"
            else:
                # No placeholders, create basic structure
                think = f"Looking at the image, I can see multiple people. Let me identify each of them systematically."
                for i in range(len(person_names)):
                    think += f"\n{{{{Identify person #{i+1}}}}}"
                enriched = self.enrich_multi_person_answer("", question, person_names)
                answer_raw = f"<think>\n{think}\n</think>\n<answer>{enriched}</answer>"

        # Extract and ENRICH <answer> section
        answer_match = re.search(r"<answer>(.*?)</answer>", answer_raw, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            enriched_answer = self.enrich_multi_person_answer(answer_content, question, person_names)
            answer_raw = re.sub(
                r"<answer>.*?</answer>",
                f"<answer>{enriched_answer}</answer>",
                answer_raw,
                flags=re.DOTALL,
            )
        else:
            # Create enriched answer if missing
            enriched = self.enrich_multi_person_answer("", question, person_names)
            answer_raw += f"\n<answer>{enriched}</answer>"

        # Rebuild to ensure clean format
        final_think_match = re.search(r"<think>(.*?)</think>", answer_raw, re.DOTALL)
        final_answer_match = re.search(r"<answer>(.*?)</answer>", answer_raw, re.DOTALL)

        if final_think_match and final_answer_match:
            think_final = final_think_match.group(1).strip()
            answer_final = final_answer_match.group(1).strip()
            answer_raw = f"<think>\n{think_final}\n</think>\n<answer>{answer_final}</answer>"

        return answer_raw, stats

    def replace_multi_tool_call_placeholders(
        self, text: str, person_nums: List[int], person_names: List[str], annotations: List[Dict]
    ) -> Tuple[str, int]:
        """
        Replace MULTIPLE {{Identify person #X}} placeholders with proper tool calls

        Returns: (fixed_text, num_replacements)
        """
        num_replacements = 0

        for person_num, person_name in zip(person_nums, person_names):
            if person_num <= len(annotations):
                bbox = annotations[person_num - 1]["bbox"]
                bbox_str = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
                replacement = f"<tool_call>Identify [{bbox_str}]</tool_call><tool_response>{person_name}</tool_response>"

                # Try exact match first
                pattern = r"\{\{Identify person #" + str(person_num) + r"\}\}"
                text, count = re.subn(pattern, replacement, text)
                num_replacements += count

                # If no exact match, replace any remaining placeholder (fallback)
                if count == 0:
                    pattern_any = r"\{\{Identify person #\d+\}\}"
                    text, count = re.subn(pattern_any, replacement, text, count=1)
                    num_replacements += count

        self.metrics["total_tool_calls"] += num_replacements
        if num_replacements >= 2:
            self.metrics["multi_tool_calls"] += 1

        return text, num_replacements

    def convert_to_conversation(self, data_entry: Dict) -> List[Dict]:
        """
        Convert one image's multi-person data to multiple conversation entries
        """
        conversations = []
        image_path = data_entry["image_path"]
        image_id = data_entry["image_id"]
        annotations = data_entry["annotations"]
        qna_list = data_entry.get("QnA_multi", [])

        for qna in qna_list:
            question = qna.get("Q", "")
            answer_raw = qna.get("A_raw", "")
            person_nums = qna.get("person_nums", [])
            person_names = qna.get("person_names", [])

            # Fix question
            question_result = self.fix_question(question)
            if question_result[0] is None:
                self.metrics["malformed_questions"] += 1
                self.metrics["removed_qna"] += 1
                continue

            question, was_malformed = question_result
            if was_malformed:
                self.metrics["malformed_questions"] += 1

            # Skip if question is too generic or broken
            if len(question.split()) < 5:
                self.metrics["removed_qna"] += 1
                continue

            # Fix and enrich answer (multi-person)
            answer_fixed, fix_stats = self.fix_answer(answer_raw, question, person_names)

            if fix_stats["missing_think"]:
                self.metrics["missing_think_tags"] += 1
            if fix_stats["missing_answer"]:
                self.metrics["missing_answer_tags"] += 1

            # Replace MULTIPLE tool call placeholders
            answer_fixed, num_replaced = self.replace_multi_tool_call_placeholders(
                answer_fixed, person_nums, person_names, annotations
            )
            self.metrics["fixed_tool_calls"] += num_replaced

            # Create conversation entry
            conversation = {
                "image": image_path,
                "image_id": image_id,
                "conversations": [
                    {"from": "system", "value": self.SYSTEM_PROMPT},
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer_fixed},
                ],
            }

            conversations.append(conversation)
            self.metrics["total_qna"] += 1

        return conversations

    def process_all(self) -> List[Dict]:
        """Process all Stage 1 Multi files and convert to final format"""
        stage1_multi_data = self.load_stage1_multi_files()

        all_conversations = []

        for data_entry in stage1_multi_data:
            self.metrics["total_images"] += 1
            conversations = self.convert_to_conversation(data_entry)
            all_conversations.extend(conversations)

        return all_conversations

    def save_results(self, conversations: List[Dict]):
        """Save final conversation array to JSON"""
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(conversations)} multi-person conversations to {self.output_file}")

    def print_metrics(self):
        """Print processing metrics"""
        print("\n" + "=" * 80)
        print("STAGE 2 MULTI METRICS")
        print("=" * 80)
        for key, value in self.metrics.items():
            print(f"{key}: {value}")
        if self.metrics["total_qna"] > 0:
            avg_tools_per_qna = self.metrics["total_tool_calls"] / self.metrics["total_qna"]
            print(f"avg_tool_calls_per_qna: {avg_tools_per_qna:.2f}")
        print("=" * 80)
