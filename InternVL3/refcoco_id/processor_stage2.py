"""
Stage 2: Refine and Convert Identity Dataset

Processes Stage 1 outputs:
1. Fixes malformed questions and answers
2. Ensures proper <think> and <answer> structure
3. Replaces {{Identify person #X}} with actual tool calls
4. **ENRICHES answers to include context from questions**
5. Converts to final conversation format with system prompt
6. Tracks metrics
"""

import json
import os
import re
from typing import Dict, List, Tuple


class IdentityStage2Processor:
    """Stage 2: Refine and convert to final format with enriched answers"""

    # System prompt (same for all conversations)
    SYSTEM_PROMPT = """You have access to an Identify tool for verifying people's identities in images. When you need to identify a person, use the tool in this format: <tool_call>Identify [x, y, x2, y2]</tool_call>

After each tool use, the person's identity will be provided as: <tool_response>Name</tool_response>. Use this capability to provide accurate identity-based responses."""

    def __init__(self, stage1_folder: str, output_file: str):
        self.stage1_folder = stage1_folder
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
        }

    def load_stage1_files(self) -> List[Dict]:
        """Load all Stage 1 JSON files"""
        json_files = []
        for file in os.listdir(self.stage1_folder):
            if file.endswith(".json"):
                filepath = os.path.join(self.stage1_folder, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    json_files.append(data)

        print(f"Loaded {len(json_files)} Stage 1 files")
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
            # Try to extract the actual question
            lines = question.split("\n")
            candidates = []

            for line in lines:
                line = line.strip()
                # Remove quotes
                line = re.sub(r'^["\']|["\']$', "", line)
                # Look for question patterns
                if line.endswith("?") and len(line.split()) >= 6:
                    candidates.append(line)

            if candidates:
                # Pick the longest/best candidate
                question = max(candidates, key=len)
            else:
                # Try to find incomplete question and complete it
                for line in lines:
                    line = line.strip()
                    if any(
                        qword in line.lower()
                        for qword in ["who is", "who are", "what is", "can you"]
                    ):
                        question = line
                        if not question.endswith("?"):
                            question += "?"
                        break
                else:
                    # No valid question found, return None to skip
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

    def extract_context_from_question(self, question: str) -> str:
        """
        Extract visual context from question for answer enrichment

        Examples:
        "Who is the person wearing the blue jacket?" -> "wearing the blue jacket"
        "Can you identify the person on the right?" -> "on the right side"
        "Who is holding the tennis racket?" -> "holding the tennis racket"
        """
        q_lower = question.lower()

        # Pattern 1: "wearing X"
        match = re.search(r"wearing (the |a |an )?([a-z\s]+?)(?:\s+in|\s+on|\s+\?|,|$)", q_lower)
        if match:
            article = match.group(1).strip() if match.group(1) else "the"
            return f"wearing {article} {match.group(2).strip()}"

        # Pattern 2: "holding X"
        match = re.search(r"holding (the |a |an )?([a-z\s]+?)(?:\s+in|\s+on|\s+\?|,|$)", q_lower)
        if match:
            article = match.group(1).strip() if match.group(1) else "the"
            return f"holding {article} {match.group(2).strip()}"

        # Pattern 3: "on the left/right"
        if "on the left" in q_lower:
            return "on the left side"
        if "on the right" in q_lower:
            return "on the right side"

        # Pattern 4: "in the center/middle"
        if "in the center" in q_lower or "in the middle" in q_lower:
            return "in the center"

        # Pattern 5: "standing/sitting at/in/near X"
        match = re.search(
            r"(standing|sitting) (?:at|in|near|on) (the )?([a-z\s]+?)(?:\?|$)", q_lower
        )
        if match:
            return f"{match.group(1)} {match.group(3).strip()}"

        # Pattern 6: "with X" (glasses, hat, etc)
        match = re.search(r"with ([a-z\s]+?)(?:\s+in|\s+on|\s+wearing|\s+\?|$)", q_lower)
        if match:
            return f"with {match.group(1).strip()}"

        # Pattern 7: "in the X shirt"
        match = re.search(r"in the ([a-z]+) (shirt|jacket|dress|coat)", q_lower)
        if match:
            return f"in the {match.group(1)} {match.group(2)}"

        return ""

    def enrich_answer(self, answer_content: str, question: str, person_name: str) -> str:
        """
        V2: Enrich answer to include context from question

        Instead of just "The person is X", create:
        "The person wearing the blue jacket is X"
        "The individual on the right side is X"
        etc.
        """
        # If answer already includes good context (>6 words), keep it
        if len(answer_content.split()) > 6:
            return answer_content

        # Extract context from question
        context = self.extract_context_from_question(question)

        if context:
            # Fix double article issue (e.g., "wearing the the blue" -> "wearing the blue")
            context = re.sub(r"\b(the|a)\s+(the|a)\b", r"\1", context)

            # Build enriched answer
            if question.lower().startswith("who are"):
                # Multiple people
                enriched = f"The people {context} are {person_name}."
            elif "individual" in question.lower():
                enriched = f"The individual {context} is {person_name}."
            else:
                enriched = f"The person {context} is {person_name}."

            self.metrics["enriched_answers"] += 1
            return enriched

        # Fallback: keep original
        return answer_content

    def fix_answer(self, answer_raw: str, question: str, person_name: str) -> Tuple[str, Dict]:
        """
        Fix answer formatting and structure
        Also enriches the <answer> section with question context
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
            # Extract placeholder and text around it
            if "{{Identify person #" in answer_raw:
                parts = re.split(r"(\{\{Identify person #\d+\}\})", answer_raw)
                before = parts[0] if len(parts) > 0 else ""
                placeholder = parts[1] if len(parts) > 1 else "{{Identify person #1}}"
                after = parts[2] if len(parts) > 2 else ""

                # Use existing content for think
                think = f"{before.strip()}\n{placeholder}\n{after.strip()}"

                # Create enriched answer
                context = self.extract_context_from_question(question)
                if context:
                    answer = f"The person {context} is {person_name}."
                else:
                    answer = f"The person is {person_name}."

                answer_raw = f"<think>\n{think}\n</think>\n<answer>{answer}</answer>"
            else:
                # No placeholder, create basic structure
                think = f"Looking at the image, I need to identify the person mentioned in the question. Let me use the Identify tool.\nBased on the identification, this is {person_name}."

                context = self.extract_context_from_question(question)
                if context:
                    answer = f"The person {context} is {person_name}."
                else:
                    answer = f"This is {person_name}."

                answer_raw = f"<think>\n{think}\n</think>\n<answer>{answer}</answer>"

        # Extract and validate <think> section
        think_match = re.search(r"<think>(.*?)</think>", answer_raw, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()

            # Ensure think is not empty
            if len(think_content.strip()) < 10:
                think_content = f"Looking at the image, I can see the person matching the description in the question. Let me identify them.\n{{{{Identify person #1}}}}\nBased on the identification, this is {person_name}."
                answer_raw = re.sub(
                    r"<think>.*?</think>",
                    f"<think>\n{think_content}\n</think>",
                    answer_raw,
                    flags=re.DOTALL,
                )

        # Extract and ENRICH <answer> section
        answer_match = re.search(r"<answer>(.*?)</answer>", answer_raw, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            # V2: Enrich with context
            enriched_answer = self.enrich_answer(answer_content, question, person_name)
            answer_raw = re.sub(
                r"<answer>.*?</answer>",
                f"<answer>{enriched_answer}</answer>",
                answer_raw,
                flags=re.DOTALL,
            )
        else:
            # Create enriched answer if missing
            context = self.extract_context_from_question(question)
            if context:
                answer_content = f"The person {context} is {person_name}."
            else:
                answer_content = f"The person is {person_name}."
            answer_raw += f"\n<answer>{answer_content}</answer>"

        # Rebuild to ensure clean format
        final_think_match = re.search(r"<think>(.*?)</think>", answer_raw, re.DOTALL)
        final_answer_match = re.search(r"<answer>(.*?)</answer>", answer_raw, re.DOTALL)

        if final_think_match and final_answer_match:
            think_final = final_think_match.group(1).strip()
            answer_final = final_answer_match.group(1).strip()
            answer_raw = f"<think>\n{think_final}\n</think>\n<answer>{answer_final}</answer>"

        return answer_raw, stats

    def replace_tool_call_placeholder(
        self, text: str, person_num: int, person_name: str, bbox: List[float]
    ) -> Tuple[str, int]:
        """
        Replace {{Identify person #X}} with proper tool call + response
        Returns: (fixed_text, num_replacements)
        """
        # Create proper tool call
        bbox_str = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
        replacement = f"<tool_call>Identify [{bbox_str}]</tool_call><tool_response>{person_name}</tool_response>"

        # Try to match the correct person number first
        pattern = r"\{\{Identify person #" + str(person_num) + r"\}\}"
        fixed_text, num_replacements = re.subn(pattern, replacement, text)

        # If no match, try to replace ANY {{Identify person #X}} placeholder
        if num_replacements == 0:
            pattern_any = r"\{\{Identify person #\d+\}\}"
            fixed_text, num_replacements = re.subn(pattern_any, replacement, text, count=1)

        return fixed_text, num_replacements

    def enrich_multi_person_answer(
        self, answer_content: str, question: str, person_names: List[str]
    ) -> str:
        """
        Enrich multi-person answer to include all identified people
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
            enriched = f"The two people are {person_names[0]} and {person_names[1]}."
        elif "left to right" in question.lower() or "from left" in question.lower():
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

    def fix_multi_person_answer(
        self, answer_raw: str, question: str, person_names: List[str]
    ) -> Tuple[str, Dict]:
        """
        Fix multi-person answer formatting (handles MULTIPLE tool call placeholders)
        """
        stats = {
            "missing_think": False,
            "missing_answer": False,
        }

        has_think = "<think>" in answer_raw and "</think>" in answer_raw
        has_answer = "<answer>" in answer_raw and "</answer>" in answer_raw

        if not has_think:
            stats["missing_think"] = True
        if not has_answer:
            stats["missing_answer"] = True

        if not has_think and not has_answer:
            think = answer_raw.strip()
            enriched = self.enrich_multi_person_answer("", question, person_names)
            answer_raw = f"<think>\n{think}\n</think>\n<answer>{enriched}</answer>"

        # Enrich <answer> section
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
            enriched = self.enrich_multi_person_answer("", question, person_names)
            answer_raw += f"\n<answer>{enriched}</answer>"

        # Rebuild format
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
        """
        num_replacements = 0

        for person_num, person_name in zip(person_nums, person_names):
            if person_num <= len(annotations):
                bbox = annotations[person_num - 1]["bbox"]
                bbox_str = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
                replacement = f"<tool_call>Identify [{bbox_str}]</tool_call><tool_response>{person_name}</tool_response>"

                pattern = r"\{\{Identify person #" + str(person_num) + r"\}\}"
                text, count = re.subn(pattern, replacement, text)
                num_replacements += count

                if count == 0:
                    pattern_any = r"\{\{Identify person #\d+\}\}"
                    text, count = re.subn(pattern_any, replacement, text, count=1)
                    num_replacements += count

        return text, num_replacements

    def convert_to_conversation(self, data_entry: Dict) -> List[Dict]:
        """
        Convert one image's data to multiple conversation entries
        (one per Q&A pair) - supports both single-person and multi-person Q&A
        """
        conversations = []
        image_path = data_entry["image_path"]
        image_id = data_entry["image_id"]
        annotations = data_entry["annotations"]
        qna_list = data_entry.get("QnA", [])
        qna_multi_list = data_entry.get("QnA_multi", [])

        # Process single-person Q&A
        for qna in qna_list:
            question = qna.get("Q", "")
            answer_raw = qna.get("A_raw", "")
            person_num = qna.get("person_num", 1)
            person_name = qna.get("person_name", "Unknown")

            # Fix question
            question_result = self.fix_question(question)
            if question_result[0] is None:
                self.metrics["malformed_questions"] += 1
                self.metrics["removed_qna"] += 1
                continue

            question, was_malformed = question_result
            if was_malformed:
                self.metrics["malformed_questions"] += 1

            if len(question.split()) < 5:
                self.metrics["removed_qna"] += 1
                continue

            # Fix and enrich answer
            answer_fixed, fix_stats = self.fix_answer(answer_raw, question, person_name)

            if fix_stats["missing_think"]:
                self.metrics["missing_think_tags"] += 1
            if fix_stats["missing_answer"]:
                self.metrics["missing_answer_tags"] += 1

            # Replace tool call placeholder
            if person_num <= len(annotations):
                bbox = annotations[person_num - 1]["bbox"]
                answer_fixed, num_replaced = self.replace_tool_call_placeholder(
                    answer_fixed, person_num, person_name, bbox
                )
                self.metrics["fixed_tool_calls"] += num_replaced

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

        # Process multi-person Q&A
        for qna in qna_multi_list:
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

            if len(question.split()) < 5:
                self.metrics["removed_qna"] += 1
                continue

            # Fix and enrich multi-person answer
            answer_fixed, fix_stats = self.fix_multi_person_answer(answer_raw, question, person_names)

            if fix_stats["missing_think"]:
                self.metrics["missing_think_tags"] += 1
            if fix_stats["missing_answer"]:
                self.metrics["missing_answer_tags"] += 1

            # Replace MULTIPLE tool call placeholders
            answer_fixed, num_replaced = self.replace_multi_tool_call_placeholders(
                answer_fixed, person_nums, person_names, annotations
            )
            self.metrics["fixed_tool_calls"] += num_replaced

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
        """Process all Stage 1 files and convert to final format"""
        stage1_data = self.load_stage1_files()

        all_conversations = []

        for data_entry in stage1_data:
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

        print(f"Saved {len(conversations)} conversations to {self.output_file}")

    def print_metrics(self):
        """Print processing metrics"""
        print("\n" + "=" * 80)
        print("STAGE 2 METRICS")
        print("=" * 80)
        for key, value in self.metrics.items():
            print(f"{key}: {value}")
        print("=" * 80)
